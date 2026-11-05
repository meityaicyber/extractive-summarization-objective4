"""
Transformer-based Cross-Encoder for sentence relevance scoring and optional MLM continued pretraining.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

from summarizer.segmentation import Unit
from summarizer.structure import severity_to_level

logger = logging.getLogger(__name__)


def position_bucket(position: float) -> str:
    """Bucket document position float into 'early', 'middle', or 'late'.

    Args:
        position: Relative position in document in [0.0, 1.0].

    Returns:
        One of 'early', 'middle', or 'late'.
    """
    if position < 0.33:
        return "early"
    if position < 0.67:
        return "middle"
    return "late"


def build_context_string(unit: Unit) -> str:
    """Build canonical first-segment context string for cross-encoder pair.

    Format: 'section: <section type>. severity: <level>. position: <early|middle|late>.'

    Args:
        unit: Extractive Unit.

    Returns:
        Structured context string.
    """
    sec = unit.section_type
    sev = severity_to_level(unit.eff_severity)
    pos = position_bucket(unit.position)
    return f"section: {sec}. severity: {sev}. position: {pos}."


class PairDataset(Dataset):
    """PyTorch Dataset for sentence relevance cross-encoder pairs."""

    def __init__(
        self,
        contexts: List[str],
        texts: List[str],
        labels: Optional[List[int]] = None,
    ):
        self.contexts = contexts
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"context": self.contexts[idx], "text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = float(self.labels[idx])
        return item


class TextDataset(Dataset):
    """PyTorch Dataset for raw texts for Masked Language Modeling."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class CrossEncoderScorer:
    """Cross-Encoder model for scoring unit relevance given contextual metadata."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        """Initialize CrossEncoderScorer.

        Args:
            model_name: HuggingFace model identifier or local path.
            device: 'cpu', 'cuda', or None (auto-detect).
            max_length: Maximum token sequence length.
        """
        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info("Initializing CrossEncoderScorer backbone '%s' on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: Optional[AutoModelForSequenceClassification] = None

    def domain_adapt(
        self,
        texts: Sequence[str],
        epochs: int = 2,
        batch_size: int = 8,
        lr: float = 5e-5,
        mlm_prob: float = 0.15,
    ) -> None:
        """Continued pretraining via Masked Language Modeling over in-domain texts.

        Note: Continued pretraining is optional (behind --domain_adapt). In domain adaptation,
        the model continues masked language modeling on in-domain audit texts before fine-tuning
        the relevance classification head. It requires enough in-domain text corpus to be useful.

        Args:
            texts: In-domain text lines or sentences.
            epochs: Number of MLM pretraining epochs.
            batch_size: Batch size for MLM training.
            lr: Learning rate for AdamW.
            mlm_prob: Probability of masking tokens.
        """
        valid_texts = [t.strip() for t in texts if len(t.strip().split()) >= 5]
        if len(valid_texts) < 10:
            logger.warning("Not enough texts for domain adaptation (%d texts); skipping.", len(valid_texts))
            return

        logger.info("Starting domain adaptation (MLM) on %d text units for %d epochs...", len(valid_texts), epochs)
        mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=mlm_prob
        )

        dataset = TextDataset(valid_texts)

        def collate_fn(batch_texts: List[str]):
            tokenized = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            return collator(
                [{"input_ids": ids} for ids in tokenized["input_ids"]]
            )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=lr)

        mlm_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                optimizer.zero_grad()
                outputs = mlm_model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(len(loader), 1)
            logger.info("Domain adaptation epoch %d/%d - loss: %.4f", epoch + 1, epochs, avg_loss)

        # Retain updated backbone weights for fine-tuning
        # Transfer the adapted backbone encoder into sequence classification
        logger.info("Domain adaptation completed. Loading adapted encoder into classification model.")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1
        ).to(self.device)

        # Copy matching encoder weights
        encoder_state = mlm_model.state_dict()
        cls_state = self.model.state_dict()
        for k, v in encoder_state.items():
            if k in cls_state and cls_state[k].shape == v.shape:
                cls_state[k] = v
        self.model.load_state_dict(cls_state)
        del mlm_model
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def fit(
        self,
        train_units: Sequence[Unit],
        train_labels: Sequence[int],
        val_units: Optional[Sequence[Unit]] = None,
        val_labels: Optional[Sequence[int]] = None,
        epochs: int = 4,
        batch_size: int = 16,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
        val_split: float = 0.15,
        seed: int = 42,
        checkpoint_dir: str = "checkpoints",
    ) -> Dict[str, Any]:
        """Fine-tune the cross-encoder with class-weighted loss and early stopping.

        Args:
            train_units: Training units.
            train_labels: Binary silver labels (0 or 1).
            val_units: Optional held-out validation units.
            val_labels: Optional held-out validation labels.
            epochs: Max training epochs (default 4).
            batch_size: Batch size (default 16).
            lr: Learning rate (default 2e-5).
            warmup_ratio: Warmup steps fraction (default 0.1).
            val_split: Fraction to hold out for validation if not provided (default 0.15).
            seed: Fixed random seed.
            checkpoint_dir: Directory where the best model checkpoint is saved.

        Returns:
            Dict of training history and best metrics.
        """
        # Set seeds for strict reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=1
            ).to(self.device)

        # Prepare context strings
        train_contexts = [build_context_string(u) for u in train_units]
        train_texts = [u.text for u in train_units]
        labels_list = [int(y) for y in train_labels]

        # Validation split
        if val_units is None or val_labels is None:
            n_total = len(train_units)
            indices = list(range(n_total))
            random.shuffle(indices)
            n_val = max(1, int(n_total * val_split))
            val_idx = set(indices[:n_val])
            train_idx = [i for i in indices if i not in val_idx]

            v_contexts = [train_contexts[i] for i in val_idx]
            v_texts = [train_texts[i] for i in val_idx]
            v_labels = [labels_list[i] for i in val_idx]

            t_contexts = [train_contexts[i] for i in train_idx]
            t_texts = [train_texts[i] for i in train_idx]
            t_labels = [labels_list[i] for i in train_idx]
        else:
            t_contexts = train_contexts
            t_texts = train_texts
            t_labels = labels_list
            v_contexts = [build_context_string(u) for u in val_units]
            v_texts = [u.text for u in val_units]
            v_labels = [int(y) for y in val_labels]

        # Calculate class weights: positives are a minority
        n_pos = sum(t_labels)
        n_neg = len(t_labels) - n_pos
        pos_weight_val = (n_neg / max(n_pos, 1))
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float, device=self.device)
        logger.info(
            "Training Cross-Encoder: %d train units (%d pos, %d neg, pos_weight=%.2f), %d val units",
            len(t_labels), n_pos, n_neg, pos_weight_val, len(v_labels)
        )

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        def collate_pairs(batch: List[Dict[str, Any]]):
            ctxs = [b["context"] for b in batch]
            txts = [b["text"] for b in batch]
            encoded = self.tokenizer(
                ctxs,
                txts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if "label" in batch[0]:
                encoded["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float)
            return encoded

        train_ds = PairDataset(t_contexts, t_texts, t_labels)
        val_ds = PairDataset(v_contexts, v_texts, v_labels)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_pairs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_pairs)

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        use_cuda_amp = (self.device == "cuda" and torch.cuda.is_available())
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

        best_val_loss = float("inf")
        best_state = None
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_checkpoint_path = os.path.join(checkpoint_dir, "best_cross_encoder")

        patience = 2
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                target_labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                    kwargs: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
                    if token_type_ids is not None:
                        kwargs["token_type_ids"] = token_type_ids
                    outputs = self.model(**kwargs)
                    logits = outputs.logits.view(-1)
                    loss = loss_fn(logits, target_labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / max(len(train_loader), 1)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    token_type_ids = batch.get("token_type_ids")
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(self.device)
                    target_labels = batch["labels"].to(self.device)

                    with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                        if token_type_ids is not None:
                            kwargs["token_type_ids"] = token_type_ids
                        outputs = self.model(**kwargs)
                        logits = outputs.logits.view(-1)
                        loss = loss_fn(logits, target_labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / max(len(val_loader), 1)
            logger.info(
                "Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f",
                epoch + 1, epochs, avg_train_loss, avg_val_loss
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                # Save best checkpoint to checkpoints/
                self.save_pretrained(best_checkpoint_path)
                logger.info("Saved new best checkpoint to %s (val_loss=%.4f)", best_checkpoint_path, best_val_loss)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return {"best_val_loss": best_val_loss, "checkpoint_path": best_checkpoint_path}

    def predict_proba(self, units: Sequence[Unit], batch_size: int = 32) -> np.ndarray:
        """Predict sentence relevance probabilities for a sequence of units.

        Args:
            units: Sequence of Unit objects.
            batch_size: Inference batch size.

        Returns:
            1D numpy array of probabilities in [0.0, 1.0].
        """
        if not units:
            return np.zeros(0, dtype=float)

        if self.model is None:
            logger.warning("Model not fitted/loaded; loading default backbone.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=1
            ).to(self.device)

        self.model.eval()
        contexts = [build_context_string(u) for u in units]
        texts = [u.text for u in units]

        probs_list: List[float] = []
        use_cuda_amp = (self.device == "cuda" and torch.cuda.is_available())

        with torch.no_grad():
            for i in range(0, len(units), batch_size):
                b_ctxs = contexts[i : i + batch_size]
                b_txts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    b_ctxs,
                    b_txts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                kwargs: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
                if "token_type_ids" in encoded:
                    kwargs["token_type_ids"] = encoded["token_type_ids"].to(self.device)

                with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                    logits = self.model(**kwargs).logits.view(-1)
                    probs = torch.sigmoid(logits).cpu().numpy()

                probs_list.extend(np.atleast_1d(probs).tolist())

        return np.asarray(probs_list, dtype=float)

    def save_pretrained(self, save_directory: str) -> None:
        """Save model weights and tokenizer to directory.

        Args:
            save_directory: Destination directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        if self.model is not None:
            self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str,
        device: Optional[str] = None,
        max_length: int = 256,
    ) -> "CrossEncoderScorer":
        """Load a saved CrossEncoderScorer checkpoint.

        Args:
            save_directory: Directory containing saved checkpoint.
            device: 'cpu', 'cuda', or None.
            max_length: Maximum sequence length.

        Returns:
            Loaded CrossEncoderScorer instance.
        """
        instance = cls.__new__(cls)
        instance.model_name = save_directory
        instance.max_length = max_length
        if device is None:
            instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            instance.device = device

        instance.tokenizer = AutoTokenizer.from_pretrained(save_directory, local_files_only=True)
        instance.model = AutoModelForSequenceClassification.from_pretrained(
            save_directory, local_files_only=True
        ).to(instance.device)
        return instance
