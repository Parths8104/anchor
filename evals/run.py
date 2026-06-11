"""CLI: run the eval harness against ./evals/cases and write a report.

Usage:
    python evals/run.py
    python evals/run.py --cases-dir custom/cases --report-path out.json
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anchor.eval.harness import EvalHarness, write_report  # noqa: E402
from anchor.logging_config import configure_logging, get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Anchor eval harness.")
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cases",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    configure_logging()
    log = get_logger(__name__)

    if not args.cases_dir.is_dir():
        log.error("cases_dir_not_found", path=str(args.cases_dir))
        return 1

    report_path = args.report_path or (
        Path(__file__).resolve().parent
        / "reports"
        / f"report-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    harness = EvalHarness()
    results = harness.run_directory(args.cases_dir)
    write_report(results, report_path)

    summary = EvalHarness.summarize(results)
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:>18}: {v}")
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
