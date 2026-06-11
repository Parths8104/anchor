"""End-to-end evaluation harness.

Loads JSON test cases, runs each through the full retrieve-then-generate
pipeline, scores the output, and writes a structured report.

Pass criteria (configurable per-case via overrides):
  - answer_similarity >= 0.78  (semantic match to reference)
  - grounded == True            (LLM-as-judge says yes)
  - citation_coverage >= 0.7    (most claims cite at least one passage)
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from anchor.eval.metrics import (
    EvalMetrics,
    citation_coverage,
    cosine_similarity,
    embed_text,
    judge_groundedness,
)
from anchor.generation.generator import Generator
from anchor.logging_config import get_logger
from anchor.retrieval.hybrid import HybridRetriever

log = get_logger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.78
DEFAULT_COVERAGE_THRESHOLD = 0.7


@dataclass(frozen=True)
class EvalCase:
    """A single eval test case loaded from JSON."""

    id: str
    question: str
    reference_answer: str
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD


@dataclass(frozen=True)
class EvalCaseResult:
    """Full record of one eval case run."""

    case_id: str
    question: str
    generated_answer: str
    reference_answer: str
    citations_count: int
    metrics: EvalMetrics


class EvalHarness:
    """Runs eval cases against a Retriever + Generator and scores results."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or Generator()

    def run_case(self, case: EvalCase) -> EvalCaseResult:
        """Run a single case and return scored result."""
        start = time.perf_counter()
        retrieval = self.retriever.retrieve(case.question)
        answer = self.generator.generate(case.question, retrieval.chunks)
        latency_ms = (time.perf_counter() - start) * 1000

        # Similarity: embed both texts and cosine-compare.
        gen_emb = embed_text(answer.answer)
        ref_emb = embed_text(case.reference_answer)
        similarity = cosine_similarity(gen_emb, ref_emb)

        grounded = judge_groundedness(answer.answer, answer.citations)
        coverage = citation_coverage(answer.answer)

        passed = (
            similarity >= case.similarity_threshold
            and grounded
            and coverage >= case.coverage_threshold
        )

        metrics = EvalMetrics(
            answer_similarity=round(similarity, 4),
            grounded=grounded,
            citation_coverage=round(coverage, 4),
            latency_ms=round(latency_ms, 2),
            passed=passed,
        )

        log.info(
            "eval_case_complete",
            case_id=case.id,
            similarity=metrics.answer_similarity,
            grounded=metrics.grounded,
            coverage=metrics.citation_coverage,
            latency_ms=metrics.latency_ms,
            passed=metrics.passed,
        )

        return EvalCaseResult(
            case_id=case.id,
            question=case.question,
            generated_answer=answer.answer,
            reference_answer=case.reference_answer,
            citations_count=len(answer.citations),
            metrics=metrics,
        )

    def run_directory(self, cases_dir: Path) -> list[EvalCaseResult]:
        """Run every *.json case file in `cases_dir`."""
        results: list[EvalCaseResult] = []
        for path in sorted(cases_dir.glob("*.json")):
            case = self._load_case(path)
            log.info("running_case", case_id=case.id, path=str(path))
            results.append(self.run_case(case))
        return results

    @staticmethod
    def _load_case(path: Path) -> EvalCase:
        data = json.loads(path.read_text(encoding="utf-8"))
        return EvalCase(
            id=data["id"],
            question=data["question"],
            reference_answer=data["reference_answer"],
            similarity_threshold=data.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD),
            coverage_threshold=data.get("coverage_threshold", DEFAULT_COVERAGE_THRESHOLD),
        )

    @staticmethod
    def summarize(results: list[EvalCaseResult]) -> dict[str, float | int]:
        """Aggregate pass-rate, mean similarity, mean latency."""
        if not results:
            return {"cases": 0, "pass_rate": 0.0, "mean_similarity": 0.0, "mean_latency_ms": 0.0}

        n = len(results)
        passed = sum(1 for r in results if r.metrics.passed)
        return {
            "cases": n,
            "passed": passed,
            "pass_rate": round(passed / n, 4),
            "mean_similarity": round(sum(r.metrics.answer_similarity for r in results) / n, 4),
            "mean_coverage": round(sum(r.metrics.citation_coverage for r in results) / n, 4),
            "grounded_rate": round(sum(1 for r in results if r.metrics.grounded) / n, 4),
            "mean_latency_ms": round(sum(r.metrics.latency_ms for r in results) / n, 2),
        }


def write_report(results: list[EvalCaseResult], path: Path) -> None:
    """Write a JSON report combining per-case results and aggregate summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "summary": EvalHarness.summarize(results),
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(report, indent=2))
    log.info("eval_report_written", path=str(path))
