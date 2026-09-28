"""Validated CPU-first neural fallback for ambiguous Core routing.

The symbolic router remains first and remains authoritative for all concrete
Knowledge operations. This optional model only predicts the broad destination
``chat`` or ``knowledge`` for inputs where the symbolic router abstains. It
never selects or executes a tool, resolves dates, resolves people, or mutates
data. Low-confidence predictions abstain and the caller falls back safely.

The network is intentionally tiny and CPU-friendly:

    signed hashed word/character features -> GELU -> 2-class softmax

The artifact is a few hundred KB to a few MB, has no tokenizer or CUDA
requirement, and can replace the 0.5B Qwen selector for broad route gating.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

LABELS = ("chat", "knowledge")
DEFAULT_FEATURES = 8192
DEFAULT_HIDDEN = 48
# A validated 98% precision target keeps the predictor useful while
# abstaining on uncertain utterances. Training may raise this value when
# validation data requires it.
DEFAULT_THRESHOLD = 0.53
DEFAULT_ARTIFACT = Path(
    os.environ.get(
        "NIX_ROUTING_PREDICTOR_PATH",
        Path(__file__).resolve().parent.parent / "data" / "routing_predictor.npz",
    )
)


def _hash_feature(value: str, dimensions: int) -> tuple[int, float]:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest[:4], "little") % dimensions
    sign = 1.0 if digest[4] & 1 else -1.0
    return bucket, sign


def _feature_vector(text: str, dimensions: int = DEFAULT_FEATURES) -> np.ndarray:
    """Create normalized signed word and character n-gram features."""
    normalized = " ".join((text or "").lower().split())
    words = normalized.split()
    vector = np.zeros(dimensions, dtype=np.float32)

    tokens = ["<s>", *words, "</s>"]
    for index, word in enumerate(tokens):
        bucket, sign = _hash_feature(f"w:{word}", dimensions)
        vector[bucket] += sign * 1.5
        if index:
            bucket, sign = _hash_feature(f"b:{tokens[index - 1]} {word}", dimensions)
            vector[bucket] += sign

    padded = f"  {normalized}  "
    for size in (2, 3, 4):
        for index in range(max(0, len(padded) - size + 1)):
            bucket, sign = _hash_feature(
                f"c:{padded[index:index + size]}", dimensions
            )
            vector[bucket] += sign * 0.35

    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _gelu(values: np.ndarray) -> np.ndarray:
    return 0.5 * values * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)
    ))


def _gelu_grad(values: np.ndarray) -> np.ndarray:
    tanh_term = np.tanh(
        np.sqrt(2.0 / np.pi) * (values + 0.044715 * values ** 3)
    )
    sech_sq = 1.0 - tanh_term ** 2
    return (
        0.5 * (1.0 + tanh_term)
        + 0.5 * values * sech_sq * np.sqrt(2.0 / np.pi)
        * (1.0 + 3.0 * 0.044715 * values ** 2)
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class TinyRoutingPredictor:
    """Small two-layer classifier with validation-selected abstention."""

    def __init__(
        self,
        *,
        dimensions: int = DEFAULT_FEATURES,
        hidden: int = DEFAULT_HIDDEN,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        self.dimensions = int(dimensions)
        self.hidden = int(hidden)
        self.threshold = float(threshold)
        self.w1 = np.zeros((self.dimensions, self.hidden), dtype=np.float32)
        self.b1 = np.zeros(self.hidden, dtype=np.float32)
        self.w2 = np.zeros((self.hidden, 2), dtype=np.float32)
        self.b2 = np.zeros(2, dtype=np.float32)
        self.ready = False
        self.validation: dict[str, float | int] = {}

    def _features(self, texts: Iterable[str]) -> np.ndarray:
        rows = [_feature_vector(text, self.dimensions) for text in texts]
        if not rows:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = features @ self.w1 + self.b1
        hidden = _gelu(z1)
        probabilities = _softmax(hidden @ self.w2 + self.b2)
        return z1, hidden, probabilities

    def _loss(self, probabilities: np.ndarray, targets: np.ndarray, class_weights: np.ndarray) -> float:
        row_weights = class_weights[np.argmax(targets, axis=1)]
        return float(-np.mean(row_weights * np.sum(
            targets * np.log(np.maximum(probabilities, 1e-8)), axis=1
        )))

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        *,
        epochs: int = 160,
        learning_rate: float = 0.045,
        weight_decay: float = 1e-4,
        validation_fraction: float = 0.2,
        patience: int = 18,
        seed: int = 42,
    ) -> dict[str, object]:
        """Train with a stratified split, class balancing, and early stop."""
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must be non-empty and equal length")
        unknown = set(labels) - set(LABELS)
        if unknown:
            raise ValueError(f"unsupported route labels: {sorted(unknown)}")

        rng = np.random.default_rng(seed)
        labels_array = np.asarray(labels)
        train_indices: list[int] = []
        validation_indices: list[int] = []
        for label in LABELS:
            indices = np.flatnonzero(labels_array == label)
            rng.shuffle(indices)
            split = max(1, int(len(indices) * validation_fraction))
            validation_indices.extend(indices[:split].tolist())
            train_indices.extend(indices[split:].tolist())
        rng.shuffle(train_indices)
        rng.shuffle(validation_indices)

        x = self._features(texts)
        targets = np.zeros((len(labels), 2), dtype=np.float32)
        for row, label in enumerate(labels):
            targets[row, LABELS.index(label)] = 1.0
        class_counts = np.bincount(np.argmax(targets[train_indices], axis=1), minlength=2)
        class_weights = len(train_indices) / np.maximum(1.0, 2.0 * class_counts)

        self.w1 = rng.normal(0.0, np.sqrt(2.0 / self.dimensions), self.w1.shape).astype(np.float32)
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden), self.w2.shape).astype(np.float32)
        self.b1.fill(0.0)
        self.b2.fill(0.0)

        best_state = None
        best_validation_loss = float("inf")
        stale = 0
        epochs_run = 0
        for epoch in range(max(1, epochs)):
            epochs_run = epoch + 1
            train = np.asarray(train_indices, dtype=np.int64)
            z1, hidden, probabilities = self._forward(x[train])
            row_weights = class_weights[np.argmax(targets[train], axis=1)][:, None]
            dlogits = row_weights * (probabilities - targets[train]) / len(train)
            dw2 = hidden.T @ dlogits + weight_decay * self.w2
            db2 = np.sum(dlogits, axis=0)
            dz1 = (dlogits @ self.w2.T) * _gelu_grad(z1)
            dw1 = x[train].T @ dz1 + weight_decay * self.w1
            db1 = np.sum(dz1, axis=0)
            self.w2 -= learning_rate * dw2.astype(np.float32)
            self.b2 -= learning_rate * db2.astype(np.float32)
            self.w1 -= learning_rate * dw1.astype(np.float32)
            self.b1 -= learning_rate * db1.astype(np.float32)

            validation = np.asarray(validation_indices, dtype=np.int64)
            _, _, validation_probabilities = self._forward(x[validation])
            validation_loss = self._loss(
                validation_probabilities,
                targets[validation],
                class_weights,
            )
            if validation_loss + 1e-5 < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = (
                    self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()
                )
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if best_state is not None:
            self.w1, self.b1, self.w2, self.b2 = best_state
        self.ready = True

        train_predictions = self.predict_many(
            [texts[i] for i in train_indices], allow_abstain=False
        )
        validation_predictions = self.predict_many(
            [texts[i] for i in validation_indices], allow_abstain=False
        )
        train_labels = [labels[i] for i in train_indices]
        validation_labels = [labels[i] for i in validation_indices]
        train_accuracy = (
            sum(a == b for a, b in zip(train_predictions, train_labels))
            / max(1, len(train_labels))
        )
        validation_accuracy = (
            sum(a == b for a, b in zip(validation_predictions, validation_labels))
            / max(1, len(validation_labels))
        )

        # Select the lowest threshold that meets the validation precision
        # gate. If no threshold meets it, abstention remains the default.
        selected_threshold = DEFAULT_THRESHOLD
        threshold_rows: list[tuple[float, float, float]] = []
        validation_probabilities = self._forward(
            x[np.asarray(validation_indices, dtype=np.int64)]
        )[2]
        for threshold in np.arange(0.50, 0.991, 0.01):
            predictions = np.argmax(validation_probabilities, axis=1)
            confidences = np.max(validation_probabilities, axis=1)
            covered = confidences >= threshold
            if not np.any(covered):
                continue
            precision = float(np.mean(
                predictions[covered] == np.argmax(
                    targets[np.asarray(validation_indices, dtype=np.int64)], axis=1
                )[covered]
            ))
            coverage = float(np.mean(covered))
            threshold_rows.append((float(threshold), precision, coverage))
        acceptable = [row for row in threshold_rows if row[1] >= 0.98]
        if acceptable:
            # Choose the lowest threshold that meets the validation precision
            # gate, but never go below the broader-corpus safety floor.
            selected_threshold = max(DEFAULT_THRESHOLD, acceptable[0][0])
        self.threshold = selected_threshold
        covered = np.max(validation_probabilities, axis=1) >= self.threshold
        covered_correct = (
            np.argmax(validation_probabilities, axis=1)[covered]
            == np.argmax(
                targets[np.asarray(validation_indices, dtype=np.int64)], axis=1
            )[covered]
        )
        self.validation = {
            "train_examples": len(train_indices),
            "validation_examples": len(validation_indices),
            "epochs": epochs_run,
            "train_accuracy": round(float(train_accuracy), 4),
            "validation_accuracy": round(float(validation_accuracy), 4),
            "threshold": round(float(self.threshold), 4),
            "validation_coverage": round(float(np.mean(covered)), 4),
            "validation_precision_when_covered": round(
                float(np.mean(covered_correct)) if len(covered_correct) else 0.0,
                4,
            ),
        }
        return dict(self.validation)

    def predict_many(self, texts: list[str], *, allow_abstain: bool = True) -> list[str]:
        if not self.ready:
            raise RuntimeError("routing predictor is not trained")
        probabilities = self._forward(self._features(texts))[2]
        output: list[str] = []
        for row in probabilities:
            confidence = float(np.max(row))
            route = LABELS[int(np.argmax(row))]
            output.append(route if not allow_abstain or confidence >= self.threshold else "unknown")
        return output

    def predict(self, text: str) -> dict[str, object]:
        if not self.ready:
            return {"route": "unknown", "confidence": 0.0, "abstained": True}
        probabilities = self._forward(self._features([text]))[2][0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        abstained = confidence < self.threshold
        return {
            "route": "unknown" if abstained else LABELS[index],
            "candidate": LABELS[index],
            "confidence": round(confidence, 4),
            "abstained": abstained,
        }

    def save(self, path: Path = DEFAULT_ARTIFACT) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            metadata=np.array(json.dumps({
                "dimensions": self.dimensions,
                "hidden": self.hidden,
                "threshold": self.threshold,
            })),
            validation=np.array(json.dumps(self.validation)),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT) -> "TinyRoutingPredictor | None":
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
                model = cls(**metadata)
                model.w1 = archive["w1"].astype(np.float32)
                model.b1 = archive["b1"].astype(np.float32)
                model.w2 = archive["w2"].astype(np.float32)
                model.b2 = archive["b2"].astype(np.float32)
                if "validation" in archive:
                    model.validation = json.loads(str(archive["validation"]))
                model.ready = True
                return model
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None


def _binary_training_examples() -> tuple[list[str], list[str]]:
    """Load balanced broad route labels from the existing corpora."""
    from adversarial_corpus import ADVERSARIAL_CORPUS
    from hard_corpus import HARD_PROMPTS
    from router_corpus import CORPUS

    examples: dict[str, str] = {}
    sources = [CORPUS, HARD_PROMPTS]
    sources.append([(text, expected) for text, expected, *_ in ADVERSARIAL_CORPUS])
    for rows in sources:
        for text, label in rows:
            if label not in LABELS:
                continue
            examples[" ".join(text.lower().split())] = label
    # Balance the broad classifier so the large Knowledge corpus does not
    # turn every uncertain utterance into a Knowledge mutation candidate.
    grouped = {label: [text for text, value in examples.items() if value == label] for label in LABELS}
    count = min(len(grouped["chat"]), len(grouped["knowledge"]))
    texts = grouped["chat"][:count] + grouped["knowledge"][:count]
    labels = ["chat"] * count + ["knowledge"] * count
    return texts, labels


def train_from_repository(*, artifact: Path = DEFAULT_ARTIFACT, epochs: int = 160) -> dict[str, object]:
    texts, labels = _binary_training_examples()
    model = TinyRoutingPredictor()
    report = model.fit(texts, labels, epochs=epochs)
    model.save(artifact)
    return {**report, "examples": len(texts), "artifact": str(artifact)}


def load_repository_predictor() -> TinyRoutingPredictor | None:
    return TinyRoutingPredictor.load()
