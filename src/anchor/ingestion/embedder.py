"""Batched embedding generation with the OpenAI API.

Why batched: OpenAI's embeddings endpoint accepts arrays. Batching gives
a 5-10x throughput improvement vs one-call-per-chunk and reduces cost
overhead from request setup.

Why retries: network blips happen. We retry on transient errors with
exponential backoff. Retries are bounded to avoid runaway loops on
permanent failures (auth, bad model name).
"""

import time
from collections.abc import Sequence

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from anchor.config import get_settings
from anchor.logging_config import get_logger

log = get_logger(__name__)

# OpenAI embedding endpoints accept up to 2048 inputs per call,
# but smaller batches give more graceful failure recovery.
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 4
INITIAL_BACKOFF_SEC = 1.0


class Embedder:
    """Generates embedding vectors for batches of text via OpenAI."""

    def __init__(self, model: str | None = None, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.batch_size = batch_size
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a sequence of texts, batching under the hood.

        Order is preserved: output[i] is the embedding of texts[i].
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for batch_start in range(0, len(texts), self.batch_size):
            batch = texts[batch_start : batch_start + self.batch_size]
            batch_embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(batch_embeddings)
            log.debug(
                "embedded_batch",
                batch_index=batch_start // self.batch_size,
                batch_size=len(batch),
            )

        return all_embeddings

    def _embed_with_retry(self, batch: Sequence[str]) -> list[list[float]]:
        """Single batch call with bounded exponential backoff retry."""
        backoff = INITIAL_BACKOFF_SEC
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.embeddings.create(model=self.model, input=list(batch))
                return [item.embedding for item in resp.data]
            except (RateLimitError, APIConnectionError) as e:
                last_error = e
                log.warning(
                    "embed_retry",
                    attempt=attempt + 1,
                    max_retries=MAX_RETRIES,
                    error=str(e),
                    backoff_sec=backoff,
                )
                time.sleep(backoff)
                backoff *= 2
            except APIError as e:
                # 4xx errors are unlikely to succeed on retry — fail fast.
                log.error("embed_permanent_error", error=str(e))
                raise

        assert last_error is not None  # for type-checker
        raise last_error
