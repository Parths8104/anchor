"""CLI: ask Anchor a question from the terminal.

Usage:
    python scripts/query.py "What is dependency injection in FastAPI?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anchor.generation.generator import Generator  # noqa: E402
from anchor.logging_config import configure_logging  # noqa: E402
from anchor.retrieval.hybrid import HybridRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask Anchor a question.")
    parser.add_argument("question", type=str, help="The question to ask.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieved chunks count.")
    args = parser.parse_args()

    configure_logging()

    retriever = HybridRetriever()
    generator = Generator()

    retrieval = retriever.retrieve(args.question, top_k=args.top_k)
    answer = generator.generate(args.question, retrieval.chunks)

    print("\n" + "=" * 72)
    print("QUESTION:", args.question)
    print("=" * 72)
    print("\nANSWER:")
    print(answer.answer)
    print("\nCITATIONS:")
    if not answer.citations:
        print("  (no citations)")
    for c in answer.citations:
        snippet = c.text[:120].replace("\n", " ")
        print(f"  [{c.index}] {c.chunk_id}: {snippet}...")
    print("\nDIAGNOSTICS:")
    print(f"  dense retrieved : {retrieval.dense_count}")
    print(f"  bm25 retrieved  : {retrieval.bm25_count}")
    print(f"  chunks used     : {len(retrieval.chunks)}")
    print(f"  prompt tokens   : {answer.prompt_tokens}")
    print(f"  completion tok. : {answer.completion_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
