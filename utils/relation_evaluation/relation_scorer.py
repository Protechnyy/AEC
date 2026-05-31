"""Micro scorers for relation extraction outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from utils.relation_data.loader import normalize_space


def tuple_key(item: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_space(str(item.get("relation_type") or item.get("relation") or "")).lower(),
        normalize_space(str(item.get("head") or "")).lower(),
        normalize_space(str(item.get("tail") or "")).lower(),
    )


def counter_prf(gold: Counter, pred: Counter) -> dict[str, float | int]:
    tp = sum((gold & pred).values())
    fp = sum((pred - gold).values())
    fn = sum((gold - pred).values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_relation_predictions(records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_triples: Counter = Counter()
    pred_triples: Counter = Counter()
    gold_labels: Counter = Counter()
    pred_labels: Counter = Counter()
    for record in records:
        for item in record.get("GoldRelations", []):
            if not isinstance(item, dict):
                continue
            key = tuple_key(item)
            if all(key):
                gold_triples[key] += 1
                gold_labels[key[0]] += 1
        for item in record.get("PredictionRelations", []):
            if not isinstance(item, dict):
                continue
            key = tuple_key(item)
            if all(key):
                pred_triples[key] += 1
                pred_labels[key[0]] += 1
    triple_scores = counter_prf(gold_triples, pred_triples)
    label_scores = counter_prf(gold_labels, pred_labels)
    return {
        "triple_precision": float(triple_scores["precision"]),
        "triple_recall": float(triple_scores["recall"]),
        "triple_f1": float(triple_scores["f1"]),
        "relation_cls_precision": float(label_scores["precision"]),
        "relation_cls_recall": float(label_scores["recall"]),
        "relation_cls_f1": float(label_scores["f1"]),
        "counts": {
            "triple": {key: int(triple_scores[key]) for key in ("tp", "fp", "fn")},
            "relation_cls": {key: int(label_scores[key]) for key in ("tp", "fp", "fn")},
        },
    }
