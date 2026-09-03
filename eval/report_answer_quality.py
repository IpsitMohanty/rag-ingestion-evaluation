"""Render a concise Markdown report from answer-quality benchmark JSON.

Usage:
    python -m eval.report_answer_quality
"""
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "results" / "answer_quality_results.json"
REPORT_PATH = REPO_ROOT / "results" / "ANSWER_QUALITY.md"


def render_report(data: dict) -> str:
    lines = [
        "# Answer-quality benchmark",
        "",
        "Generated from `results/answer_quality_results.json`.",
        "",
        "| Variant | Correctness | Faithfulness | Abstention accuracy | Mean latency | Answered |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in data.items():
        summary = value["summary"]
        lines.append(
            f"| {name} | {summary['correctness_rate']:.1%} | "
            f"{summary['faithfulness_rate']:.1%} | "
            f"{summary['abstention_accuracy']:.1%} | "
            f"{summary['mean_latency_seconds']:.2f}s | "
            f"{summary['answered']}/{summary['n']} |"
        )

    lines.extend(["", "## Incorrect answered queries", ""])
    for name, value in data.items():
        incorrect = [
            result["query_id"]
            for result in value["results"]
            if not result["abstained"] and result["correct"] is False
        ]
        lines.append(f"- **{name}:** {', '.join(incorrect) or 'none'}")

    lines.extend(["", "## Abstained queries", ""])
    for name, value in data.items():
        abstained = [result["query_id"] for result in value["results"] if result["abstained"]]
        lines.append(f"- **{name}:** {', '.join(abstained) or 'none'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Benchmark results not found: {RESULTS_PATH}")
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    REPORT_PATH.write_text(render_report(data), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
