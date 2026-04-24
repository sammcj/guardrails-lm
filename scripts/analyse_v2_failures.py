"""One-off: analyse v2's OOD failure modes.

Runs v2 against:
  - deepset/prompt-injections test split (attack-labelled) -> find false negatives
  - leolee99/NotInject NotInject_three (all benign)        -> find false positives

Then does light pattern analysis and writes `reference/failure-analysis-v2.md`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset

from guardrails.config import Settings
from guardrails.model import load_for_inference
from guardrails.ood import _score

REPORT_PATH = Path("reference/failure-analysis-v2.md")
CHECKPOINT = Path("checkpoints/best")

TRIGGER_WORDS = [
    "ignore",
    "override",
    "system",
    "prompt",
    "previous",
    "instructions",
    "reveal",
    "bypass",
    "jailbreak",
    "DAN",
    "pretend",
    "role",
    "roleplay",
    "developer",
    "admin",
    "root",
    "sudo",
    "reset",
    "disregard",
    "forget",
    "rule",
    "rules",
    "filter",
    "restriction",
    "guideline",
]

TOKENISE = re.compile(r"[A-Za-z]+")


def length_buckets(lengths: list[int]) -> dict[str, int]:
    buckets = {"<50": 0, "50-150": 0, "150-400": 0, "400-1000": 0, ">1000": 0}
    for n in lengths:
        if n < 50:
            buckets["<50"] += 1
        elif n < 150:
            buckets["50-150"] += 1
        elif n < 400:
            buckets["150-400"] += 1
        elif n < 1000:
            buckets["400-1000"] += 1
        else:
            buckets[">1000"] += 1
    return buckets


def trigger_counts(texts: list[str]) -> dict[str, int]:
    all_lower = " ".join(texts).lower()
    return {w: all_lower.count(w) for w in TRIGGER_WORDS}


def top_ngrams(texts: list[str], n: int = 2, k: int = 20) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for text in texts:
        words = TOKENISE.findall(text.lower())
        for i in range(len(words) - n + 1):
            counter[" ".join(words[i : i + n])] += 1
    return counter.most_common(k)


def fmt_sample(text: str, prob: float, max_chars: int = 240) -> str:
    clean = " ".join(text.split())
    if len(clean) > max_chars:
        clean = clean[:max_chars] + "..."
    return f"- p={prob:.3f} | {clean}"


def _split_by_prediction(
    texts: list[str], probs: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    preds = (probs >= threshold).astype(int)
    fn_idx = np.where((labels == 1) & (preds == 0))[0]
    tp_idx = np.where((labels == 1) & (preds == 1))[0]
    texts_by = {
        "fn": [texts[i] for i in fn_idx],
        "tp": [texts[i] for i in tp_idx],
    }
    probs_by = {
        "fn": [float(probs[i]) for i in fn_idx],
        "tp": [float(probs[i]) for i in tp_idx],
    }
    return texts_by, probs_by


def _render_section(
    title: str,
    intro: list[str],
    length_header: tuple[str, str],
    trigger_header: tuple[str, str],
    bigrams_header: str,
    examples_header: str,
    wrong: list[str],
    right: list[str],
    wrong_probs: list[float],
    wrong_first: bool,
) -> list[str]:
    md: list[str] = [f"## {title}", ""]
    md.extend(intro)
    md.append("")
    md.append(
        f"### Length distribution (chars)\n\n| bucket | {length_header[0]} | {length_header[1]} |"
    )
    md.append("| --- | --- | --- |")
    wl = length_buckets([len(t) for t in wrong])
    rl = length_buckets([len(t) for t in right])
    for b in wl:
        md.append(f"| {b} | {wl[b]} | {rl[b]} |")
    md.append("")
    md.append(f"### Trigger-word frequency\n\n| word | {trigger_header[0]} | {trigger_header[1]} |")
    md.append("| --- | --- | --- |")
    wt, rt = trigger_counts(wrong), trigger_counts(right)
    for w in TRIGGER_WORDS:
        if wt[w] + rt[w] > 0:
            md.append(f"| {w} | {wt[w]} | {rt[w]} |")
    md.append("")
    md.append(f"### {bigrams_header}\n")
    for bg, c in top_ngrams(wrong, n=2, k=15):
        md.append(f"- `{bg}` ({c})")
    md.append("")
    md.append(f"### {examples_header}\n")
    paired = sorted(zip(wrong_probs, wrong, strict=True), reverse=wrong_first)
    for p, t in paired[:20]:
        md.append(fmt_sample(t, p))
    md.append("")
    return md


def main() -> None:
    settings = Settings()
    threshold = json.loads((CHECKPOINT / "threshold.json").read_text())["threshold"]
    print(f"v2 threshold: {threshold:.4f}")
    tokenizer, model = load_for_inference(settings, CHECKPOINT)

    print("Loading deepset/prompt-injections test split...")
    ds1 = load_dataset("deepset/prompt-injections", split="test")
    t1 = list(ds1["text"])
    p1 = _score(tokenizer, model, t1, settings)
    l1 = np.array(ds1["label"], dtype=int)
    g1, gp1 = _split_by_prediction(t1, p1, l1, threshold)
    print(
        f"deepset attacks: {int((l1 == 1).sum())}, caught: {len(g1['tp'])}, missed: {len(g1['fn'])}"
    )

    print("Loading NotInject_three...")
    ds2 = load_dataset("leolee99/NotInject", split="NotInject_three")
    t2 = list(ds2["prompt"])
    p2 = _score(tokenizer, model, t2, settings)
    preds2 = (p2 >= threshold).astype(int)
    fp_texts = [t2[i] for i in np.where(preds2 == 1)[0]]
    fp_probs = [float(p2[i]) for i in np.where(preds2 == 1)[0]]
    tn_texts = [t2[i] for i in np.where(preds2 == 0)[0]]
    print(f"NotInject_three: n={len(t2)}, flagged (FPs): {len(fp_texts)}")

    md: list[str] = [
        "# v2 failure-mode analysis",
        "",
        f"Model: `{CHECKPOINT}` | threshold: {threshold:.4f}",
        "",
        "Purpose: before trying more augmentation, understand WHY v2 misses deepset attacks "
        "and over-flags NotInject benign prompts. Patterns here inform whether LLM-as-judge, "
        "targeted data, or a different loss function is the right next move.",
        "",
    ]

    n_attacks = int((l1 == 1).sum())
    md += _render_section(
        "1. deepset false negatives (attacks v2 missed)",
        [
            f"- Attack prompts in split: **{n_attacks}**. Caught: {len(g1['tp'])}. "
            f"**Missed: {len(g1['fn'])}** ({len(g1['fn']) / max(1, n_attacks):.0%}).",
            f"- Mean probability on misses: **{float(np.mean(gp1['fn'])):.3f}** "
            "(i.e. model is mostly confident they're benign, not on the fence).",
        ],
        ("missed (FN)", "caught (TP)"),
        ("in missed", "in caught"),
        "Top bigrams in missed attacks",
        "20 most-confidently-wrong examples",
        g1["fn"],
        g1["tp"],
        gp1["fn"],
        wrong_first=False,  # lowest prob = most confidently wrong for FN
    )

    md += _render_section(
        "2. NotInject_three false positives (benign v2 flagged)",
        [
            f"- Benign prompts: {len(t2)}. "
            f"**Flagged as attacks: {len(fp_texts)}** ({len(fp_texts) / max(1, len(t2)):.0%}).",
            f"- Mean probability on FPs: **{float(np.mean(fp_probs)) if fp_probs else 0:.3f}**.",
        ],
        ("flagged (FP)", "passed (TN)"),
        ("in flagged", "in passed"),
        "Top bigrams in flagged benign",
        "20 most-confidently-wrong examples",
        fp_texts,
        tn_texts,
        fp_probs,
        wrong_first=True,  # highest prob = most confidently wrong for FP
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(md))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
