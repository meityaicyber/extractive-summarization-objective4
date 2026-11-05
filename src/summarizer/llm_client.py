"""
Local Gemma client. Loads the model from a folder on disk with Hugging Face
transformers and generates in-process. No server or API is involved.

Interface used by the rest of the code:
    client.chat(user, system="", max_new_tokens=None) -> str
    client.ping() -> str
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional, Tuple

logger = logging.getLogger("gemma")

# Gemma turn markers. In this tokenizer they are plain text, not special tokens,
# so generation is stopped by checking the generated text for them.
GEMMA_TURN_START = "<start_of_turn>"
GEMMA_TURN_END = "<end_of_turn>"
GEMMA_STOP_STRINGS = (GEMMA_TURN_END, GEMMA_TURN_START)


class GemmaError(RuntimeError):
    """Raised when the model cannot load or produces no usable output."""


def _cut_at_markers(text: str) -> str:
    """Remove everything from the first turn marker onward, then trim whitespace."""
    cut = len(text)
    for marker in GEMMA_STOP_STRINGS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


class GemmaClient:
    """Loads a local Gemma model once and serves chat-style generation calls.

    Calls are serialized with a lock, since one model instance on one GPU
    does not benefit from running several generations at once.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_new_tokens: int = 2048,
        max_input_tokens: int = 16384,
        temperature: float = 0.0,
        load_in_4bit: bool = False,
    ) -> None:
        path = model_path or os.environ.get("GEMMA_MODEL_PATH", "")
        if not path:
            raise GemmaError("No model path. Set GEMMA_MODEL_PATH or pass --model_path.")
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(os.path.join(path, "config.json")):
            raise GemmaError(f"config.json not found in {path}.")

        self.model_path = path
        self.name = os.path.basename(path.rstrip(os.sep))
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.temperature = temperature
        self._lock = threading.Lock()
        self.tokenizer, self.model = self._load(path, load_in_4bit)

    @staticmethod
    def _load(path: str, load_in_4bit: bool) -> Tuple[Any, Any]:
        """Load the tokenizer and model from a local folder."""
        try:
            import torch
            import transformers
            from transformers import AutoTokenizer
        except ImportError as e:
            raise GemmaError(
                "Missing dependency. Run: pip install -U torch transformers accelerate"
            ) from e

        kwargs: dict = {"device_map": "auto", "torch_dtype": "auto"}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise GemmaError("4-bit loading needs: pip install bitsandbytes") from e
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        tokenizer = AutoTokenizer.from_pretrained(path)

        errors = []
        started = time.time()
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                errors.append(f"{cls_name}: not available in this transformers version")
                continue
            try:
                model = cls.from_pretrained(path, **kwargs)
                model.eval()
                logger.info("Model loaded with %s in %.0fs", cls_name, time.time() - started)
                return tokenizer, model
            except Exception as e:
                errors.append(f"{cls_name}: {type(e).__name__}: {str(e)[:300]}")

        raise GemmaError("Could not load the model:\n  " + "\n  ".join(errors))

    def _build_prompt(self, content: str) -> str:
        """Format one user message for the model.

        Uses the tokenizer's own chat template when the model folder has one.
        Otherwise uses the Gemma turn format directly.
        """
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return (
            f"{self.tokenizer.bos_token or '<bos>'}"
            f"{GEMMA_TURN_START}user\n{content}{GEMMA_TURN_END}\n"
            f"{GEMMA_TURN_START}model\n"
        )

    def _stopping_criteria(self, start_len: int):
        """Stop as soon as the newly generated text contains a turn marker."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        import torch

        tokenizer = self.tokenizer

        class _StopOnMarker(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                new_text = tokenizer.decode(input_ids[0, start_len:], skip_special_tokens=False)
                done = any(m in new_text for m in GEMMA_STOP_STRINGS)
                return torch.tensor([done], device=input_ids.device)

        return StoppingCriteriaList([_StopOnMarker()])

    def chat(
        self,
        user: str,
        system: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a reply to one prompt.

        Args:
            user: User message content.
            system: Optional instructions, placed in the same user turn because
                some Gemma chat templates reject a separate system role.
            max_new_tokens: Override the reply length for this call.

        Returns:
            The reply text, cut at the first turn marker.

        Raises:
            GemmaError: If the prompt is too long, GPU memory runs out, or the
                model returns nothing.
        """
        import torch

        content = f"{system}\n\n{user}" if system else user

        with self._lock:
            prompt = self._build_prompt(content)
            # The prompt already starts with BOS, so the tokenizer must not add another.
            enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            n_in = int(enc["input_ids"].shape[1])
            if n_in > self.max_input_tokens:
                raise GemmaError(
                    f"Prompt is {n_in} tokens, above max_input_tokens={self.max_input_tokens}. "
                    "Raise --max_input_tokens if GPU memory allows, or shorten the input."
                )

            enc = enc.to(self.model.device)
            gen_kwargs: dict = {
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "stopping_criteria": self._stopping_criteria(n_in),
            }
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
            if self.tokenizer.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

            try:
                with torch.inference_mode():
                    out = self.model.generate(**enc, **gen_kwargs)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                raise GemmaError(
                    "CUDA out of memory. Lower --max_input_tokens, or run with --load_in_4bit."
                ) from e

        new_ids = out[0, n_in:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        text = _cut_at_markers(raw)
        if not text:
            raise GemmaError("Model returned empty output.")
        return text

    def ping(self) -> str:
        """Run one short generation to confirm the model works."""
        return self.chat("Reply with the single word OK.", max_new_tokens=16)


class OllamaClient:
    """Ollama API client for models like Qwen and Nemotron.

    Shares the same .chat(...) and .ping() signature as GemmaClient.
    """

    def __init__(
        self,
        model_name: str,
        host: str = "http://localhost:11434",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.name = model_name
        self.host = host.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._lock = threading.Lock()

    def chat(
        self,
        user: str,
        system: str = "",
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate a reply using Ollama."""
        import json
        import re
        import urllib.request
        import urllib.error

        content = user
        payload = {
            "model": self.model_name,
            "prompt": content,
            "system": system,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens or self.max_new_tokens,
                "temperature": self.temperature,
            },
        }

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self._lock:
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.URLError as e:
                raise GemmaError(f"Ollama connection error: {e}") from e

        text = data.get("response", "").strip()
        # Remove any thinking blocks if generated by reasoning models
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        text = _cut_at_markers(text)
        if not text:
            raise GemmaError(f"Ollama model {self.model_name} returned empty output.")
        return text

    def ping(self) -> str:
        """Run one short generation to confirm the model works."""
        return self.chat("Reply with the single word OK.", max_new_tokens=16)


def get_llm_client(model_target: str, **kwargs: Any) -> Any:
    """Instantiate an LLM client based on target name or path.

    If model_target is a directory or points to local weights, loads GemmaClient.
    Otherwise connects via OllamaClient.
    """
    if os.path.isdir(model_target) or (os.path.isfile(os.path.join(model_target, "config.json")) if os.path.exists(model_target) else False):
        return GemmaClient(model_path=model_target, **kwargs)
    return OllamaClient(model_name=model_target, **kwargs)
