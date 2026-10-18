"""The comparison table: the three original decoders are required, and the
retrieval rows show up only when their result files exist.
"""

from __future__ import annotations

import json

from vision2words.evaluation.report import build_comparison_table, build_metrics_json


def _scores(cider: float, latency: float | None = None) -> dict:
    m = {"bleu1": 0.6, "bleu2": 0.4, "bleu3": 0.3, "bleu4": 0.2, "cider": cider}
    if latency is not None:
        m["latency_ms_per_image"] = latency
    return m


def _decoder_result(model_type: str, cider: float) -> dict:
    return {
        "model_type": model_type,
        "num_params": 1000,
        "mean_train_epoch_seconds": 10.0,
        "greedy": _scores(cider, 5.0),
        "beam": _scores(cider + 0.02, 12.0),
    }


def _write(results_dir, name, data):
    (results_dir / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def _base(results_dir):
    _write(results_dir, "lstm_baseline", _decoder_result("lstm", 0.40))
    _write(results_dir, "lstm_attention", _decoder_result("lstm_attention", 0.45))
    _write(results_dir, "transformer", _decoder_result("transformer", 0.47))


def test_table_without_retrieval_results_has_only_the_original_decoders(tmp_path):
    _base(tmp_path)
    combined = build_metrics_json(tmp_path)
    build_comparison_table(combined, tmp_path)
    table = (tmp_path / "comparison.md").read_text(encoding="utf-8")

    assert set(combined) == {"lstm_baseline", "lstm_attention", "transformer"}
    assert "retrieval" not in table.lower()


def test_retrieval_results_are_added_when_present(tmp_path):
    _base(tmp_path)
    _write(tmp_path, "transformer_rag", _decoder_result("transformer_rag", 0.52))
    _write(
        tmp_path,
        "rag_llm",
        {
            "model_type": "rag_llm",
            "llm": "Qwen/Qwen2.5-0.5B-Instruct",
            "retrieval": {"k": 5, "captions_per_neighbor": 2, "index_size": 6000},
            "retrieval_only": _scores(0.35),
            "rag_llm": {**_scores(0.38, 40.0), "batch_size": 8, "device": "cuda"},
        },
    )
    combined = build_metrics_json(tmp_path)
    build_comparison_table(combined, tmp_path)
    table = (tmp_path / "comparison.md").read_text(encoding="utf-8")

    assert "| Transformer + retrieval |" in table
    assert "## Retrieval + language model" in table
    assert "| Retrieval + Qwen2.5-0.5B-Instruct |" in table
    assert "| Retrieval only (consensus caption) |" in table
    assert json.loads((tmp_path / "metrics.json").read_text())["rag_llm"]["llm"].startswith("Qwen")
