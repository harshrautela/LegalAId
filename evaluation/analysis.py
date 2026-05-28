from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import RESULTS_DIR

SUMMARY_CSV = RESULTS_DIR / "evaluation_summary.csv"
SUMMARY_JSON = RESULTS_DIR / "evaluation_summary.json"
FAILURE_CSV = RESULTS_DIR / "failure_cases.csv"
FAILURE_JSONL = RESULTS_DIR / "failure_cases" / "failure_cases.jsonl"
CHART_DIR = RESULTS_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_markdown_report(df: pd.DataFrame, out_path: Path):
    lines = []
    lines.append("# LegalAId Evaluation Report")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append(df.to_markdown(index=False))
    lines.append("")
    lines.append("## Notes")
    lines.append("- `rougeL` is mainly used for summarization and QA.")
    lines.append("- `exact_match` and `f1` are mainly for QA.")
    lines.append("- `groundedness` and `hallucination_rate` are proxy metrics.")
    lines.append("- `recall@5` and `mrr` are retrieval metrics.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)
    print("\n=== Evaluation Summary ===")
    print(df.to_string(index=False))

    # Save a cleaned report
    report_path = RESULTS_DIR / "evaluation_report.csv"
    df.to_csv(report_path, index=False)
    logger.info(f"Saved report → {report_path}")

    # Save markdown report
    md_report = RESULTS_DIR / "evaluation_report.md"
    save_markdown_report(df, md_report)
    logger.info(f"Saved markdown report → {md_report}")

    # Charts by metric
    metric_cols = ["rouge1", "rouge2", "rougeL", "bleu", "exact_match", "f1", "groundedness", "recall_at_5", "mrr"]
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        sub = df[["task", "model", metric]].dropna()
        if sub.empty:
            continue

        fig = px.bar(
            sub,
            x="model",
            y=metric,
            color="task",
            barmode="group",
            title=f"LegalAId – {metric} by model and task",
        )
        out = CHART_DIR / f"{metric}.html"
        fig.write_html(out)
        logger.info(f"Saved chart → {out}")

    # Failure cases
    failures = load_jsonl(FAILURE_JSONL)
    if failures:
        fail_df = pd.DataFrame(failures)
        fail_out = RESULTS_DIR / "failure_cases.csv"
        fail_df.to_csv(fail_out, index=False)
        logger.info(f"Saved failure cases CSV → {fail_out}")
        print("\n=== Sample Failure Cases ===")
        print(fail_df.head(10).to_string(index=False))
    elif FAILURE_CSV.exists():
        fail_df = pd.read_csv(FAILURE_CSV)
        print("\n=== Sample Failure Cases ===")
        print(fail_df.head(10).to_string(index=False))

    # Save a small text summary
    summary_txt = RESULTS_DIR / "evaluation_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("LegalAId Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n")
    logger.info(f"Saved text summary → {summary_txt}")

    print("\nAnalysis complete ✓")


if __name__ == "__main__":
    main()