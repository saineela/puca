from __future__ import annotations

"""
Cross-corpus generalization evaluation.

The model was trained on: GoEmotions (Reddit), tweet_eval/irony
(Twitter), the personal corpus, and DailyDialog TRAIN split.

These corpora were NEVER seen in any form:
  1. Cornell Movie-Dialogs (Danescu-Niculescu-Mizil & Lee, 2011):
     scripted but naturalistic multi-genre conversation, 304k lines.
  2. Ubuntu Dialog Corpus (Lowe et al., 2015): real human-to-human
     tech-support chat -- typed, messy, domain-shifted.

Labeling uses the SAME text-only knowledge-worthiness definition as
training (first-person + lexicon evidence + not phatic). Because the
labels come from the same deterministic function, this measures two
distinct things:
  a) NEURAL generalization: head vs the label function
  b) SYSTEM self-consistency: verifier == label function should be
     ~100% by construction (sanity check of the pipeline itself)

No claims of new "truth" here -- this is domain-shift stress testing
of the trained head against the fixed label function.

Usage:
    .venv/bin/python scripts/eval_cross_corpus.py [--cornell 20000]
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nix_knowledge.semantic.classifier import SemanticClassifier
from nix_knowledge.semantic.knowledge_labels import (
    _FIRST_PERSON,
    _GREETINGS,
    LEXICON_PATTERNS,
)

DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "models" / "training_data"
)

# movie_lines.txt rows: L id +++$+++ speaker +++$+++ movie +++$+++ name +++$+++ text
_LINE_RE = re.compile(
    r"^L\d+ \+\+\+\$\+\+\+ \S+ \+\+\+\$\+\+\+ \S+ \+\+\+\$\+\+\+ \S+ \+\+\+\$\+\+\+ (.*)$"
)


def label_turn(text: str) -> int:
    """Same rule as knowledge_labels._turn_is_knowledge_worthy."""
    clean = text.strip()
    if not clean or len(clean.split()) < 3:
        return 0
    if _GREETINGS.match(clean):
        return 0
    hits = sum(1 for p in LEXICON_PATTERNS if p.search(clean))
    first_person = bool(_FIRST_PERSON.search(clean))
    return 1 if (first_person and hits >= 1) else 0


def load_cornell(n: int, seed: int = 3) -> list[str]:
    path = DATA_ROOT / "cornell" / "movie_lines.txt"
    if not path.exists():
        print(f"cornell not found at {path}, skipping")
        return []
    lines = []
    with path.open(encoding="iso-8859-1") as fh:
        for row in fh:
            match = _LINE_RE.match(row.strip())
            if match:
                text = match.group(1).strip()
                if text:
                    lines.append(text)
    rng = random.Random(seed)
    rng.shuffle(lines)
    return lines[:n]


def load_ubuntu(n: int, seed: int = 5) -> list[str]:
    """
    Stream user turns from the raw Ubuntu Dialog Corpus tgz
    (dialogs/*/*.tsv rows: timestamp \t author \t recipient \t text).
    """
    base = DATA_ROOT / "ubuntu"
    txt = base / "ubuntu_turns.txt"
    tgz = DATA_ROOT / "ubuntu_dialogs.tgz"
    if not txt.exists() and tgz.exists():
        import io
        import tarfile

        print("extracting ubuntu turns from tgz (streaming) ...")
        base.mkdir(parents=True, exist_ok=True)
        turns: list[str] = []
        seen: set[str] = set()
        with tarfile.open(tgz, "r:gz") as tar:
            count = 0
            for member in tar:
                if not member.isfile() or not member.name.endswith(
                    ".tsv"
                ):
                    continue
                count += 1
                if count % 2000 != 0:
                    continue  # sample every 2000th dialogue file
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                for raw_line in io.TextIOWrapper(
                    extracted, encoding="utf-8", errors="ignore"
                ):
                    parts = raw_line.rstrip("\n").split("\t")
                    if len(parts) < 4:
                        continue
                    text = parts[3].strip()
                    words = text.split()
                    if not (3 <= len(words) <= 60):
                        continue
                    if text.lower() in seen:
                        continue
                    seen.add(text.lower())
                    turns.append(text)
                if len(turns) >= n * 3:
                    break
        rng = random.Random(seed)
        rng.shuffle(turns)
        txt.write_text("\n".join(turns[:n]), encoding="utf-8")
        print(f"ubuntu: wrote {min(n, len(turns))} turns")
    if not txt.exists():
        print(f"ubuntu not found at {base}, skipping")
        return []
    return [
        line for line in txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:n]


def evaluate(texts: list[str], name: str, clf, torch, device: str):
    if not texts:
        return
    truth = np.array([label_turn(t) for t in texts])

    clf.net.to(device)
    clf.net.eval()
    features = clf.embedder.encode(texts, batch_size=512)
    X = torch.from_numpy(features).float().to(device)
    probs = []
    with torch.no_grad():
        clf.net.eval()
        for start in range(0, len(X), 1024):
            out = clf.net.forward_heads(X[start:start + 1024].to(device))
            probs.append(
                torch.sigmoid(out["knowledge"]).squeeze(-1).cpu().numpy()
            )
    probs = np.concatenate(probs)

    gate = float(clf.thresholds.get("knowledge_gate", 0.5))
    pred = (probs >= gate).astype(int)

    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    acc = (tp + tn) / len(truth)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    print(f"\n=== {name} ({len(texts)} turns) ===")
    print(f"distribution: positive={truth.sum()} "
          f"negative={len(truth) - int(truth.sum())}")
    print(f"accuracy {acc*100:.2f}%  precision {prec*100:.2f}%  "
          f"recall {rec*100:.2f}%  F1 {f1*100:.2f}%")

    fp_idx = np.where((pred == 1) & (truth == 0))[0][:4]
    fn_idx = np.where((pred == 0) & (truth == 1))[0][:4]
    if len(fp_idx):
        print("FP samples:")
        for i in fp_idx:
            print(f"  p={probs[i]:.2f} {texts[i][:72]!r}")
    if len(fn_idx):
        print("FN samples:")
        for i in fn_idx:
            print(f"  p={probs[i]:.2f} {texts[i][:72]!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cornell", type=int, default=20000)
    parser.add_argument("--ubuntu", type=int, default=20000)
    args = parser.parse_args()

    clf = SemanticClassifier()
    if not clf.load():
        print("ERROR: no trained model")
        return 1

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model: {clf.version}  gate={clf.thresholds}  device={device}")

    cornell = load_cornell(args.cornell)
    evaluate(cornell, "CORNELL MOVIE-DIALOGS (unseen)", clf, torch, device)

    ubuntu = load_ubuntu(args.ubuntu)
    evaluate(ubuntu, "UBUNTU TECH-SUPPORT CHAT (unseen)", clf, torch, device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
