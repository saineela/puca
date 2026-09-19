from __future__ import annotations

"""
SemanticClassifier: multi-head neural classifier over bge embeddings.

Architecture
------------
  text -> frozen bge-small-en-v1.5 (384-d) -> per-head MLP:
      emotion      multi-label sigmoid   (GoEmotions 28 labels)
      tone         4-way softmax         (formal/casual/urgent/neutral)
      importance   scalar sigmoid        (0..1 salience)
      category     8-way softmax         (fact/preference/event/...)

The backbone is frozen (feature extractor); only the heads train, so
training is minutes on GPU and the artifact is tiny (<1MB).

Usage
-----
    clf = SemanticClassifier()
    clf.load()                     # or train() / train_from_dataset()
    result = clf.classify("I hate when my meetings get moved")
    # -> {"emotion": {...}, "tone": {...}, "importance": 0.72, ...}
"""

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import Embedder

_MODEL_DIR = (
    Path(__file__).resolve().parent.parent.parent / "models"
)
ARTIFACT_DIR = _MODEL_DIR / "semantic_head"

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "realization", "relief", "remorse", "sadness", "surprise",
    "neutral",
]

TONE_LABELS = ["formal", "casual", "urgent", "neutral"]

CATEGORY_LABELS = [
    "personal_fact",    # "I live in Austin"
    "preference",       # "I love spicy food"
    "event",            # "My wedding is June 3rd"
    "relationship",     # "Sarah is my sister"
    "skill",            # "I know Python"
    "opinion",          # "Vim is better than Emacs"
    "goal",             # "I want to run a marathon"
    "general",          # everything else
]

HEADS = ("emotion", "tone", "importance", "category")


def _torch():
    import torch

    return torch


def _nn():
    import torch.nn as nn

    return nn


class MultiHeadNet:
    """Tiny MLP stack: one hidden layer per head, trained jointly."""

    def __init__(self, input_dim: int):
        nn = _nn()

        def block(out: int) -> "nn.Module":
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, out),
            )

        self.torch_nn = nn
        self.emotion = block(len(EMOTION_LABELS))
        self.tone = block(len(TONE_LABELS))
        self.importance = block(1)
        self.category = block(len(CATEGORY_LABELS))
        self.sarcasm = block(1)
        # deeper head for knowledge-worthiness: the task is a subtle
        # lexical-boundary function that a single hidden layer cannot
        # capture from frozen embeddings alone.
        self.knowledge = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        self.modules = [
            self.emotion, self.tone, self.importance, self.category,
            self.sarcasm, self.knowledge,
        ]
        # (name, module) pairs -- names are the persistence keys, so
        # each head's weights survive save/load independently even
        # though all heads share the same nn.Sequential class.
        self.named = [
            ("emotion", self.emotion),
            ("tone", self.tone),
            ("importance", self.importance),
            ("category", self.category),
            ("sarcasm", self.sarcasm),
            ("knowledge", self.knowledge),
        ]

    def parameters(self):
        for m in self.modules:
            yield from m.parameters()

    def to(self, device: str) -> None:
        for m in self.modules:
            m.to(device)

    def train(self, mode: bool = True) -> None:
        for m in self.modules:
            m.train(mode)

    def eval(self) -> None:
        self.train(False)

    def forward_heads(self, features):
        return {
            "emotion": self.emotion(features),
            "tone": self.tone(features),
            "importance": self.importance(features),
            "category": self.category(features),
            "sarcasm": self.sarcasm(features),
            "knowledge": self.knowledge(features),
        }


class SemanticClassifier:
    """
    High-level classifier: embed -> predict all heads.

    Thread-safe; the torch forward pass is serialized by a lock.
    """

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.net: MultiHeadNet | None = None
        self.version: str = "untrained"
        self.thresholds: dict[str, float] = {}
        self._lock = threading.Lock()
        self._load_meta()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def weights_path(self) -> Path:
        return ARTIFACT_DIR / "heads.pt"

    @property
    def meta_path(self) -> Path:
        return ARTIFACT_DIR / "meta.json"

    def _ensure_dirs(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        if self.net is None:
            raise RuntimeError("nothing to save: model is untrained")
        self._ensure_dirs()
        torch = _torch()
        torch.save(
            {name: m.state_dict() for name, m in self.net.named},
            self.weights_path,
        )
        self._save_meta()

    def load(self) -> bool:
        torch = _torch()
        if not self.weights_path.exists():
            return False
        self._load_meta()
        input_dim = self.embedder.dimension
        self.net = MultiHeadNet(input_dim)
        state = torch.load(self.weights_path, map_location="cpu")
        by_name = {name: m for name, m in self.net.named}
        for name, sd in state.items():
            if name in by_name:
                by_name[name].load_state_dict(sd)
        self.net.eval()
        return True

    def _load_meta(self) -> None:
        if not self.meta_path.exists():
            return
        try:
            meta = json.loads(self.meta_path.read_text())
            self.version = meta.get("version", self.version)
            self.thresholds = meta.get("thresholds", {})
        except (json.JSONDecodeError, OSError):
            pass

    def _save_meta(self) -> None:
        self._ensure_dirs()
        self.meta_path.write_text(
            json.dumps(
                {
                    "version": self.version,
                    "thresholds": self.thresholds,
                    "input_dim": self.embedder.dimension,
                    "labels": {
                        "emotion": EMOTION_LABELS,
                        "tone": TONE_LABELS,
                        "category": CATEGORY_LABELS,
                    },
                },
                indent=2,
            )
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, text: str) -> dict[str, Any]:
        """
        Classify one text. Returns:
          {
            "emotion":    [(label, prob), ... top 5],
            "tone":       (label, prob),
            "importance": float,
            "category":   (label, prob),
            "confidence": float   # min head confidence, calibrated-ish
          }
        """
        if self.net is None and not self.load():
            return self._heuristic_fallback(text)

        torch = _torch()
        vector = self.embedder.encode_one(text)

        with self._lock, torch.no_grad():
            self.net.eval()
            features = torch.from_numpy(vector).float().unsqueeze(0)
            outputs = self.net.forward_heads(features)

        emotion_probs = torch.sigmoid(outputs["emotion"])[0].numpy()
        tone_probs = torch.softmax(outputs["tone"], dim=-1)[0].numpy()
        importance = float(
            torch.sigmoid(outputs["importance"])[0].item()
        )
        category_probs = torch.softmax(
            outputs["category"], dim=-1
        )[0].numpy()

        threshold = self.thresholds.get("emotion", 0.35)
        emotions = sorted(
            (
                (EMOTION_LABELS[i], float(p))
                for i, p in enumerate(emotion_probs)
                if p >= threshold
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        if not emotions:
            i = int(np.argmax(emotion_probs))
            emotions = [(EMOTION_LABELS[i], float(emotion_probs[i]))]

        tone_i = int(np.argmax(tone_probs))
        category_i = int(np.argmax(category_probs))

        tone_conf = float(tone_probs[tone_i])
        category_conf = float(category_probs[category_i])
        sarcasm_prob = float(
            torch.sigmoid(outputs["sarcasm"]).item()
        )
        knowledge_prob = float(
            torch.sigmoid(outputs["knowledge"]).item()
        )
        # calibrated gate threshold (set by training calibration on
        # literal in-domain statements; raw 0.5 reflects tweet prior)
        sarcasm_gate = float(self.thresholds.get("sarcasm_gate", 0.8))
        knowledge_gate = float(self.thresholds.get("knowledge_gate", 0.5))

        return {
            "emotion": emotions,
            "tone": (TONE_LABELS[tone_i], tone_conf),
            "importance": round(importance, 4),
            "category": (CATEGORY_LABELS[category_i], category_conf),
            "sarcasm": (
                sarcasm_prob >= sarcasm_gate, round(sarcasm_prob, 4)
            ),
            "literal": sarcasm_prob < sarcasm_gate,
            "knowledge_worthy": knowledge_prob >= knowledge_gate,
            "knowledge_probability": round(knowledge_prob, 4),
            "storage_eligible": (
                sarcasm_prob < sarcasm_gate
                and knowledge_prob >= knowledge_gate
            ),
            "confidence": round(
                min(tone_conf, category_conf, max(importance, 0.3)),
                4,
            ),
        }

    def _heuristic_fallback(self, text: str) -> dict[str, Any]:
        """Used only when no trained artifact exists yet."""
        lower = text.lower()
        exclam = text.count("!")
        urgent_words = ("now", "asap", "urgent", "immediately", "today")
        tone = "neutral"
        if any(w in lower for w in urgent_words) or exclam >= 2:
            tone = "urgent"
        elif exclam == 1 or any(
            w in lower for w in ("hey", "yeah", "kinda", "stuff")
        ):
            tone = "casual"
        importance = 0.5
        if re_first_person(lower):
            importance = 0.75
        return {
            "emotion": [("neutral", 0.5)],
            "tone": (tone, 0.4),
            "importance": importance,
            "category": ("general", 0.34),
            "sarcasm": (False, 0.0),
            "literal": True,
            "knowledge_worthy": True,
            "knowledge_probability": 0.5,
            "storage_eligible": True,
            "confidence": 0.34,
            "fallback": True,
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _fit(
        self,
        features: np.ndarray,
        targets: dict[str, tuple],
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str,
        val_split: float,
        tag: str,
        loss_fns: dict,
    ) -> dict[str, Any]:
        """
        Shared training loop for a subset of heads.

        targets: head -> (tensor, converter) where converter maps raw
                 per-head outputs to the loss inputs.
        """
        torch = _torch()
        nn = _nn()
        from sklearn.metrics import f1_score

        X = torch.from_numpy(features).float()
        n = len(X)
        idx = np.random.RandomState(42).permutation(n)
        n_val = max(1, int(n * val_split))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        self.net.train()
        modules = [
            m for name, m in self.net.named if name in targets
        ]
        opt = torch.optim.AdamW(
            (p for m in modules for p in m.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        history: list[float] = []
        best_val = float("inf")
        best_state = None

        def batch_loss(batch) -> torch.Tensor:
            out = self.net.forward_heads(X[batch].to(device))
            total = 0.0
            for head, (y, _) in targets.items():
                raw = out[head]
                spec = loss_fns[head]
                pos_weight = None
                if isinstance(spec, tuple):
                    spec, pos_weight = spec
                if spec == "bce":
                    target = y[batch].to(device)
                    if raw.dim() != target.dim() and raw.size(-1) == 1:
                        raw = raw.squeeze(-1)
                    criterion = (
                        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                        if pos_weight is not None
                        else nn.BCEWithLogitsLoss()
                    )
                    total = total + criterion(raw, target)
                elif spec == "ce":
                    total = total + nn.CrossEntropyLoss()(
                        raw, y[batch].to(device)
                    )
                else:  # mse
                    total = total + nn.MSELoss()(
                        raw.squeeze(-1), y[batch].to(device)
                    )
            return total

        for epoch in range(epochs):
            perm = np.random.RandomState(epoch).permutation(
                len(train_idx)
            )
            epoch_loss = 0.0
            batches = 0
            for start in range(0, len(perm), batch_size):
                batch = train_idx[perm[start:start + batch_size]]
                opt.zero_grad()
                loss = batch_loss(batch)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item())
                batches += 1

            history.append(epoch_loss / max(1, batches))

            with torch.no_grad():
                self.net.eval()
                val_loss = float(
                    batch_loss(val_idx).item()
                )
                self.net.train()

            if val_loss < best_val:
                best_val = val_loss
                best_state = [
                    {k: v.detach().cpu().clone()
                     for k, v in m.state_dict().items()}
                    for m in modules
                ]
            print(
                f"[{tag}] epoch {epoch + 1}/{epochs}  "
                f"loss {history[-1]:.4f}  val {val_loss:.4f}"
            )

        if best_state is not None:
            for m, sd in zip(modules, best_state):
                m.load_state_dict(sd)
        self.net.eval()

        return {
            "final_loss": history[-1] if history else None,
            "best_val": best_val,
            "train_examples": len(train_idx),
            "val_examples": n_val,
        }

    def train_emotion_head(
        self,
        texts: list[str],
        emotion_y: np.ndarray,
        *,
        epochs: int = 6,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        device: str | None = None,
        val_split: float = 0.05,
    ) -> dict[str, Any]:
        """Supervised multi-label training of the emotion head."""
        from sklearn.metrics import f1_score

        torch = _torch()
        device = device or (
            "cuda" if _torch().cuda.is_available() else "cpu"
        )
        if self.net is None:
            self.net = MultiHeadNet(self.embedder.dimension)
        self.net.to(device)

        features = self.embedder.encode(
            texts, batch_size=512, show_progress=True
        )
        y = torch.from_numpy(emotion_y).float()

        metrics = self._fit(
            features,
            {"emotion": (y, None)},
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            val_split=val_split,
            tag="emotion",
            loss_fns={"emotion": "bce"},
        )

        # threshold tuning on the validation split
        X = torch.from_numpy(features).float()
        n_val = metrics["val_examples"]
        with torch.no_grad():
            out = self.net.forward_heads(X[:n_val].to(device))
            probs = torch.sigmoid(out["emotion"]).cpu().numpy()
        y_val = emotion_y[:n_val]
        best_t, best_f1 = 0.35, -1.0
        for t in np.arange(0.15, 0.65, 0.05):
            f1 = f1_score(
                y_val, (probs >= t).astype(int),
                average="micro", zero_division=0,
            )
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.thresholds["emotion"] = round(best_t, 2)
        self.thresholds["emotion_f1"] = round(best_f1, 4)
        metrics["emotion_f1"] = self.thresholds["emotion_f1"]
        return metrics

    def train_sarcasm_head(
        self,
        texts: list[str],
        y: np.ndarray,
        *,
        epochs: int = 5,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        device: str | None = None,
        val_split: float = 0.1,
    ) -> dict[str, Any]:
        """Binary sarcasm head: y in {0.0 = literal, 1.0 = sarcastic}."""
        from sklearn.metrics import accuracy_score

        torch = _torch()
        device = device or (
            "cuda" if _torch().cuda.is_available() else "cpu"
        )
        if self.net is None:
            self.net = MultiHeadNet(self.embedder.dimension)
        self.net.to(device)

        features = self.embedder.encode(
            texts, batch_size=512, show_progress=True
        )

        metrics = self._fit(
            features,
            {"sarcasm": (torch.from_numpy(y).float(), None)},
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            val_split=val_split,
            tag="sarcasm",
            loss_fns={"sarcasm": "bce"},
        )

        X = torch.from_numpy(features).float()
        n_val = metrics["val_examples"]
        with torch.no_grad():
            out = self.net.forward_heads(X[:n_val].to(device))
            probs = torch.sigmoid(out["sarcasm"]).squeeze(-1).cpu().numpy()
        metrics["sarcasm_accuracy"] = round(
            accuracy_score(y[:n_val], (probs >= 0.5).astype(int)), 4
        )
        return metrics

    def train_knowledge_head(
        self,
        texts: list[str],
        y: np.ndarray,
        *,
        epochs: int = 4,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        device: str | None = None,
        val_split: float = 0.05,
        fine_tune_backbone: bool = True,
        backbone_layers: int = 2,
        backbone_lr: float = 2e-5,
    ) -> dict[str, Any]:
        """
        Binary knowledge-worthiness head.
        Trains ONLY on the provided (training) split; evaluation
        against a held-out split happens elsewhere, never here.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
        )

        torch = _torch()
        device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.net is None:
            self.net = MultiHeadNet(self.embedder.dimension)
        self.net.to(device)

        features = self.embedder.encode(
            texts, batch_size=512, show_progress=True
        )

        # Optional backbone fine-tuning: unfreeze the top transformer
        # layers so the encoder itself learns the lexical boundary.
        # TRAIN-SPLIT ONLY. Features are then recomputed per batch.
        backbone_params = []
        if fine_tune_backbone:
            encoder = self.embedder.model[0].auto_model
            for param in encoder.parameters():
                param.requires_grad = False
            layers = getattr(encoder, "encoder", None)
            layers = getattr(layers, "layer", None)
            if layers is not None and len(layers) >= backbone_layers:
                for layer in list(layers)[-backbone_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                        backbone_params.append(param)

        # class imbalance: ~15% positive -> weight positives
        pos_weight = torch.tensor(
            [(y == 0).sum() / max(1, (y == 1).sum())],
            device=device,
        )
        metrics = self._fit(
            features,
            {"knowledge": (torch.from_numpy(y).float(), None)},
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            val_split=val_split,
            tag="knowledge",
            loss_fns={"knowledge": ("bce", pos_weight)},
        )

        # End-to-end fine-tuning of the top backbone layers plus the
        # knowledge head. TRAIN-SPLIT ONLY. A fresh pristine embedder
        # is used so repeated runs never drift; production resolves to
        # the persisted fine-tuned backbone via save_finetuned().
        if fine_tune_backbone and backbone_params:
            nn = _nn()
            from torch.utils.data import TensorDataset, DataLoader

            print(
                f"[knowledge] fine-tuning top {backbone_layers} "
                "backbone layers end-to-end..."
            )
            train_ft = self.embedder.model
            train_ft.to(device)

            head_modules = [
                m for name, m in self.net.named if name == "knowledge"
            ]
            head_params = [
                p for m in head_modules for p in m.parameters()
            ]
            opt = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": backbone_lr},
                    {"params": head_params, "lr": learning_rate},
                ],
                weight_decay=1e-4,
            )
            pos_weight = torch.tensor(
                [(y == 0).sum() / max(1, (y == 1).sum())], device=device
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            idx_all = np.arange(len(texts))
            rs = np.random.RandomState(42)
            rs.shuffle(idx_all)
            n_val_ft = max(1, int(len(idx_all) * val_split))
            val_idx_ft, tr_idx_ft = idx_all[:n_val_ft], idx_all[n_val_ft:]

            for epoch in range(max(2, epochs // 2)):
                train_ft.train()
                self.net.train()
                perm = rs.permutation(len(tr_idx_ft))
                ep_loss = 0.0
                batches = 0
                for start in range(0, len(perm), 64):
                    b_idx = tr_idx_ft[perm[start:start + 64]]
                    b_texts = [texts[i] for i in b_idx]
                    b_labels = torch.from_numpy(
                        y[b_idx]
                    ).float().to(device)
                    # direct forward (NOT .encode()) so autograd can
                    # reach the unfrozen backbone layers; run the FULL
                    # pipeline (transformer -> pooling -> normalize)
                    transformer = train_ft[0]
                    enc = transformer.tokenizer(
                        b_texts,
                        padding=True,
                        truncation=True,
                        max_length=256,
                        return_tensors="pt",
                    )
                    enc = {
                        k: v.to(device) for k, v in enc.items()
                    }
                    embeddings = train_ft(enc)[
                        "sentence_embedding"
                    ]
                    raw = self.net.knowledge(embeddings).squeeze(-1)
                    loss = criterion(raw, b_labels)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    ep_loss += float(loss.item())
                    batches += 1

                # quick val on train-split slice
                train_ft.eval()
                self.net.eval()
                v_probs = []
                with torch.no_grad():
                    for start in range(0, len(val_idx_ft), 256):
                        v_idx = val_idx_ft[start:start + 256]
                        v_texts = [texts[i] for i in v_idx]
                        transformer = train_ft[0]
                        enc = transformer.tokenizer(
                            v_texts,
                            padding=True,
                            truncation=True,
                            max_length=256,
                            return_tensors="pt",
                        )
                        enc = {
                            k: v.to(device)
                            for k, v in enc.items()
                        }
                        emb = train_ft(enc)["sentence_embedding"]
                        v_probs.append(
                            torch.sigmoid(
                                self.net.knowledge(emb).squeeze(-1)
                            ).cpu().numpy()
                        )
                v_probs = np.concatenate(v_probs)
                v_pred = (v_probs >= 0.5).astype(int)
                v_f1 = f1_score(
                    y[val_idx_ft].astype(int), v_pred, zero_division=0
                )
                print(
                    f"[knowledge-ft] epoch {epoch + 1} "
                    f"loss {ep_loss / max(1, batches):.4f} "
                    f"train-val F1 {v_f1:.4f}"
                )

            # persist the fine-tuned backbone for production
            out_dir = self.embedder.save_finetuned()
            print(f"[knowledge-ft] fine-tuned backbone saved: {out_dir}")
            # re-freeze backbone params (leave artifacts clean)
            for p in backbone_params:
                p.requires_grad = False

        # in-training metrics on the validation SLICE OF THE TRAIN SPLIT
        X = torch.from_numpy(features).float()
        n_val = metrics["val_examples"]
        with torch.no_grad():
            out = self.net.forward_heads(X[:n_val].to(device))
            probs = torch.sigmoid(out["knowledge"]).squeeze(-1).cpu().numpy()

        # calibrate the decision threshold for best F1 on TRAIN data
        best_t, best_f1 = 0.5, -1.0
        truth_train = y[:n_val].astype(int)
        for t in np.arange(0.30, 0.75, 0.05):
            f1 = f1_score(truth_train, (probs >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.thresholds["knowledge_gate"] = round(best_t, 2)

        pred = (probs >= best_t).astype(int)
        truth = y[:n_val].astype(int)
        metrics["val_accuracy"] = round(accuracy_score(truth, pred), 4)
        metrics["val_precision"] = round(
            precision_score(truth, pred, zero_division=0), 4
        )
        metrics["val_recall"] = round(
            recall_score(truth, pred, zero_division=0), 4
        )
        metrics["val_f1"] = round(
            f1_score(truth, pred, zero_division=0), 4
        )
        return metrics

    def train_attribute_heads(
        self,
        texts: list[str],
        tone_y: np.ndarray,
        importance_y: np.ndarray,
        category_y: np.ndarray,
        *,
        epochs: int = 30,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str | None = None,
        val_split: float = 0.1,
    ) -> dict[str, Any]:
        """Supervised training of tone/category (CE) + importance (MSE)."""
        from sklearn.metrics import accuracy_score

        torch = _torch()
        device = device or (
            "cuda" if _torch().cuda.is_available() else "cpu"
        )
        if self.net is None:
            self.net = MultiHeadNet(self.embedder.dimension)
        self.net.to(device)

        features = self.embedder.encode(
            texts, batch_size=512, show_progress=True
        )

        metrics = self._fit(
            features,
            {
                "tone": (torch.from_numpy(tone_y).long(), None),
                "importance": (
                    torch.from_numpy(importance_y).float(), None
                ),
                "category": (
                    torch.from_numpy(category_y).long(), None
                ),
            },
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            val_split=val_split,
            tag="attributes",
            loss_fns={"tone": "ce", "importance": "mse",
                      "category": "ce"},
        )

        # quick accuracy readout on the validation split
        X = torch.from_numpy(features).float()
        n_val = metrics["val_examples"]
        with torch.no_grad():
            out = self.net.forward_heads(X[:n_val].to(device))
        tone_pred = torch.softmax(out["tone"], -1).argmax(-1).cpu().numpy()
        cat_pred = torch.softmax(out["category"], -1).argmax(-1).cpu().numpy()
        imp_pred = torch.sigmoid(out["importance"]).squeeze(-1).cpu().numpy()
        metrics["tone_accuracy"] = round(
            accuracy_score(tone_y[:n_val], tone_pred), 4
        )
        metrics["category_accuracy"] = round(
            accuracy_score(category_y[:n_val], cat_pred), 4
        )
        metrics["importance_mae"] = round(
            float(np.abs(importance_y[:n_val] - imp_pred).mean()), 4
        )
        return metrics


def re_first_person(lower_text: str) -> bool:
    for marker in (" i ", "my ", "me ", "i'", "i "):
        if marker in f" {lower_text} ":
            return True
    return False
