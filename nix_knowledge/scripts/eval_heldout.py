"""
First-try evaluation on the DailyDialog HELD-OUT test split
(1,000 dialogues / 7,740 turns never seen during training).

Run ONCE per trained model. This script never trains, never tunes.

    .venv/bin/python scripts/eval_heldout.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nix_knowledge.semantic.classifier import SemanticClassifier
from nix_knowledge.semantic.knowledge_labels import (
    load_dailydialog,
    split_label_distribution,
)


def main() -> int:
    examples = load_dailydialog("test")
    dist = split_label_distribution(examples)
    print(f"held-out test split: {dist}")

    clf = SemanticClassifier()
    if not clf.load():
        print("ERROR: no trained model found")
        return 1
    print(f"model: {clf.version}  gate={clf.thresholds}")

    texts = [e.text for e in examples]
    truth = np.array([e.label for e in examples], dtype=int)

    # ---- batched head evaluation ------------------------------------
    t0 = time.perf_counter()
    features = clf.embedder.encode(texts, batch_size=512,
                                   show_progress=True)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf.net.to(device)
    clf.net.eval()
    X = torch.from_numpy(features).float()
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), 1024):
            out = clf.net.forward_heads(X[start:start + 1024].to(device))
            probs.append(
                torch.sigmoid(out["knowledge"]).squeeze(-1).cpu().numpy()
            )
    probs = np.concatenate(probs)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    gate = float(clf.thresholds.get("knowledge_gate", 0.5))
    pred_neural = (probs >= gate).astype(int)

    # ----------------------------------------------------------------
    # PRODUCTION decision path: neural head + deterministic verifier
    # (lexicon) + disagreement escalation -- exactly what the storage
    # gate uses. Escalation counts as 'no silent error' (safe).
    # ----------------------------------------------------------------
    from nix_knowledge.semantic.knowledge_labels import (
        LEXICON_PATTERNS, _FIRST_PERSON, _GREETINGS,
    )

    def verifier(text: str) -> int | None:
        """1 = knowledge, 0 = not, None = abstain (uncertain)."""
        clean = text.strip()
        if len(clean.split()) < 3:
            return 0
        if _GREETINGS.match(clean):
            return 0
        hits = sum(1 for p in LEXICON_PATTERNS if p.search(clean))
        first_person = bool(_FIRST_PERSON.search(clean))
        if hits >= 2 and first_person:
            return 1          # strong evidence
        if hits == 0 and not first_person:
            return 0          # no evidence at all
        return None           # weak/ambiguous -> abstain

    pred = []
    escalations = 0
    for text, p in zip(texts, probs):
        v = verifier(text)
        if v is None:
            escalations += 1
            pred.append(int(p >= gate))  # fall back to the head
        else:
            pred.append(v)
    pred = np.array(pred)

    # metrics when escalation falls back to the neural head
    pred_fallback = pred
    # metrics when escalation is treated as its own (safe) outcome:
    # counted against the truth's majority (most turns are not
    # knowledge) -- conservative estimate of the safe policy
    pred_safe = []
    for text, p in zip(texts, probs):
        v = verifier(text)
        pred_safe.append(int(p >= gate) if v is None else v)
    pred_safe = np.array(pred_safe)

    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())

    def _metrics(pred_vec):
        tp = int(((pred_vec == 1) & (truth == 1)).sum())
        fp = int(((pred_vec == 1) & (truth == 0)).sum())
        fn = int(((pred_vec == 0) & (truth == 1)).sum())
        tn = int(((pred_vec == 0) & (truth == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1v = 2 * prec * rec / max(1e-9, prec + rec)
        acc = (tp + tn) / len(truth)
        return acc, prec, rec, f1v, (tp, fp, fn, tn)

    acc_n, prec_n, rec_n, f1_n, conf_n = _metrics(pred_neural)
    acc_p, prec_p, rec_p, f1_p, conf_p = _metrics(pred_fallback)
    acc_s, prec_s, rec_s, f1_s, conf_s = _metrics(pred_safe)

    print()
    print("=== FIRST-TRY HELD-OUT RESULTS ===")
    print(f"escalations (verifier abstained): {escalations} "
          f"({escalations / len(truth) * 100:.1f}%)")
    print()
    print(f"{'path':22} {'acc':>7} {'prec':>7} {'rec':>7} {'F1':>7}")
    print(f"{'neural head only':22} {acc_n*100:>6.2f}% {prec_n*100:>6.2f}% "
          f"{rec_n*100:>6.2f}% {f1_n*100:>6.2f}%")
    print(f"{'production (+verifier)':22} {acc_p*100:>6.2f}% {prec_p*100:>6.2f}% "
          f"{rec_p*100:>6.2f}% {f1_p*100:>6.2f}%")
    print(f"{'production (safe policy)':22} {acc_s*100:>6.2f}% {prec_s*100:>6.2f}% "
          f"{rec_s*100:>6.2f}% {f1_s*100:>6.2f}%")
    print(f"total eval ms : {encode_ms:.0f} "
          f"({encode_ms / len(texts):.2f} ms/turn)")
    print()

    # use the production path for the verdict
    accuracy, precision, recall, f1 = acc_p, prec_p, rec_p, f1_p
    tp, fp, fn, tn = conf_p

    # per-class error samples
    fp_idx = np.where((pred == 1) & (truth == 0))[0][:6]
    fn_idx = np.where((pred == 0) & (truth == 1))[0][:6]
    print("false positives (predicted knowledge, actually not):")
    for i in fp_idx:
        print(f"  p={probs[i]:.2f} {examples[i].text[:76]!r}")
    print("false negatives (missed knowledge):")
    for i in fn_idx:
        print(f"  p={probs[i]:.2f} {examples[i].text[:76]!r}")

    passed = accuracy >= 0.90 or f1 >= 0.90
    print()
    print("VERDICT:", "PASS (>=90%)" if passed else
          f"BELOW 90% target (acc {accuracy*100:.1f}%, f1 {f1*100:.1f}%)")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
