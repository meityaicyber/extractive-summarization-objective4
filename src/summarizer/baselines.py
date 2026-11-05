"""
Baseline summarizers (Lead-k, TextRank, LexRank, Gradient-Boosting) and embedding utilities.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from summarizer.segmentation import Unit
from summarizer.structure import SECTION_TYPES

logger = logging.getLogger(__name__)

RISK_WORDS = (
    "vulnerab", "exploit", "unauthori", "injection", "exposure", "non-complian",
    "noncomplian", "fail", "weak", "insecure", "misconfig", "risk", "breach", "missing"
)


class SentenceEmbedder:
    """Computes dense sentence embeddings and cosine similarity matrix.

    Supports transformer models (e.g. sentence-transformers) or falls back
    to TF-IDF projected to LSA space for 100% offline local execution.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        lsa_dims: int = 128,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.lsa_dims = lsa_dims
        self.random_state = random_state
        self.st_model = None

        if model_name:
            try:
                from sentence_transformers import SentenceTransformer
                self.st_model = SentenceTransformer(model_name)
                logger.info("SentenceEmbedder using sentence-transformer '%s'", model_name)
            except Exception as e:
                logger.warning("Could not load sentence-transformer '%s' (%s); using LSA fallback.", model_name, e)

    @property
    def shared_space(self) -> bool:
        """True if using a model with consistent global vector space."""
        return self.st_model is not None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode list of texts into normalized embedding matrix.

        Args:
            texts: Sequence of strings.

        Returns:
            2D numpy array of shape (N, D).
        """
        n = len(texts)
        if n == 0:
            return np.zeros((0, 1), dtype=float)

        if self.st_model is not None:
            emb = self.st_model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(emb, dtype=float)

        try:
            tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
            X = tfidf.fit_transform(texts)
        except ValueError:
            return np.zeros((n, 1), dtype=float)

        k = min(self.lsa_dims, X.shape[0] - 1, X.shape[1] - 1)
        if k < 1:
            return np.zeros((n, 1), dtype=float)

        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        Z = svd.fit_transform(X)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return Z / norms

    def similarity_matrix(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute embeddings and pairwise cosine similarity matrix.

        Args:
            texts: Sequence of text strings.

        Returns:
            Tuple of (embeddings array, similarity matrix array).
        """
        n = len(texts)
        if n == 0:
            return np.zeros((0, 1)), np.zeros((0, 0))
        emb = self.encode(texts)
        sim = emb @ emb.T
        return emb, sim


def build_handcrafted_features(
    units: Sequence[Unit],
    emb: np.ndarray,
    sim: np.ndarray,
    shared_space: bool = False,
) -> np.ndarray:
    """Extract structural, severity, and similarity features for GBDT baseline.

    Args:
        units: Sequence of Unit objects.
        emb: Embedding array (N, D).
        sim: Pairwise cosine similarity matrix (N, N).
        shared_space: Whether embedding space is globally aligned.

    Returns:
        Feature matrix X of shape (N, num_features).
    """
    n = len(units)
    if n == 0:
        return np.zeros((0, 1), dtype=float)

    centroid = emb.mean(axis=0)
    cn = np.linalg.norm(centroid)
    if cn > 0:
        centroid = centroid / cn
    centroid_sim = emb @ centroid

    if n > 1:
        s = sim.astype(float).copy()
        np.fill_diagonal(s, np.nan)
        mean_other = np.nanmean(s, axis=1)
        max_other = np.nanmax(s, axis=1)
    else:
        mean_other = np.zeros(n)
        max_other = np.zeros(n)

    rows = []
    for i, u in enumerate(units):
        t = u.text.lower()
        onehot = [1.0 if u.section_type == st else 0.0 for st in SECTION_TYPES]
        risk = sum(t.count(w) for w in RISK_WORDS)
        digit_ratio = sum(c.isdigit() for c in u.text) / max(len(u.text), 1)

        row = onehot + [
            u.position,
            u.section_pos,
            float(u.is_table_row),
            min(u.words, 80) / 80.0,
            u.severity,
            u.eff_severity,
            float(u.finding_id is not None),
            min(risk, 5) / 5.0,
            digit_ratio,
            float(centroid_sim[i]),
            float(mean_other[i]),
            float(max_other[i]),
        ]
        rows.append(row)

    X = np.nan_to_num(np.asarray(rows, dtype=float))
    if shared_space:
        X = np.hstack([X, emb])
    return X


class GradientBoostingScorer:
    """Handcrafted feature-based Gradient Boosting scorer baseline."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        )
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingScorer":
        """Fit GBDT model on feature matrix and labels."""
        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            logger.warning("GBDT training labels have single class; cannot fit classifier.")
            self.fitted = False
            return self

        self.model.fit(X, y_arr)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities."""
        if not self.fitted:
            return np.full(X.shape[0], 0.5, dtype=float)
        return self.model.predict_proba(X)[:, 1]


def lead_scores(units: Sequence[Unit]) -> np.ndarray:
    """Generate Lead-k baseline scores (earlier units receive higher scores).

    Args:
        units: Sequence of Unit objects.

    Returns:
        1D array of scores in [0.0, 1.0].
    """
    n = len(units)
    if n == 0:
        return np.zeros(0, dtype=float)
    return np.array([1.0 - u.position for u in units], dtype=float)


def _tfidf_sim(texts: Sequence[str]) -> np.ndarray:
    n = len(texts)
    if n < 2:
        return np.zeros((n, n), dtype=float)
    try:
        X = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(texts)
        return (X @ X.T).toarray()
    except ValueError:
        return np.zeros((n, n), dtype=float)


def textrank_scores(units: Sequence[Unit], damping: float = 0.85) -> np.ndarray:
    """Compute TextRank scores (PageRank centrality on sentence TF-IDF graph).

    Args:
        units: Sequence of Unit objects.
        damping: PageRank damping factor.

    Returns:
        1D array of centrality scores.
    """
    texts = [u.text for u in units]
    n = len(texts)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    S = _tfidf_sim(texts)
    np.fill_diagonal(S, 0.0)
    G = nx.from_numpy_array(S)
    try:
        pr = nx.pagerank(G, alpha=damping, weight="weight")
        return np.array([pr[i] for i in range(n)], dtype=float)
    except Exception as e:
        logger.warning("TextRank PageRank failed (%s); using uniform scores.", e)
        return np.full(n, 1.0 / n, dtype=float)


def lexrank_scores(units: Sequence[Unit], threshold: float = 0.1) -> np.ndarray:
    """Compute LexRank scores (eigenvector / PageRank centrality on thresholded cosine graph).

    Args:
        units: Sequence of Unit objects.
        threshold: Minimum cosine similarity threshold.

    Returns:
        1D array of centrality scores.
    """
    texts = [u.text for u in units]
    n = len(texts)
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    S = _tfidf_sim(texts)
    A = np.where(S >= threshold, S, 0.0)
    np.fill_diagonal(A, 0.0)
    G = nx.from_numpy_array(A)
    try:
        c = nx.eigenvector_centrality_numpy(G, weight="weight")
        return np.array([c[i] for i in range(n)], dtype=float)
    except Exception:
        try:
            c = nx.pagerank(G, weight="weight")
            return np.array([c[i] for i in range(n)], dtype=float)
        except Exception as e:
            logger.warning("LexRank centrality failed (%s); using uniform scores.", e)
            return np.full(n, 1.0 / n, dtype=float)
