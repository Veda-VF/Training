# Embedding Pipeline 
# install dependencies:
"""
python -m pip install openai langchain-text-splitters pypdf tiktoken
Set your API key in PowerShell rather than hard-coding it in the script:
$env:OPENAI_API_KEY = "your-api-key"

The pipeline uses:
- pypdf to read PDFs.
- langchain-text-splitters for RecursiveCharacterTextSplitter.
- openai to generate embeddings.
- sqlite3 for persistent embedding cache.
- tiktoken for more realistic token and cost estimates.
- JSONL output with deterministic point IDs and payloads for Qdrant.
"""

import os
import json
import time
import uuid
import hashlib
import logging
import sqlite3
from pathlib import Path

import tiktoken
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =====================================================
# 1. LOGGING CONFIGURATION
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("embedding_pipeline.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


# =====================================================
# 2. EMBEDDING PIPELINE
# =====================================================

class EmbeddingPipeline:
    """
    Read documents, split text, generate and cache embeddings,
    and export points compatible with Qdrant's point format.
    """

    def __init__(
        self,
        input_dir,
        output_file="qdrant_points.jsonl",
        cache_file="embedding_cache.db",
        model="text-embedding-3-small",
        batch_size=64,
        chunk_size=1000,
        chunk_overlap=150,
        max_retries=5,
        requests_per_minute=60,
        price_per_million_tokens=0.02,
        dimensions=None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")

        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )

        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")

        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1.")

        if price_per_million_tokens < 0:
            raise ValueError("Token price cannot be negative.")

        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        self.cache_file = Path(cache_file)

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.requests_per_minute = requests_per_minute
        self.price_per_million_tokens = price_per_million_tokens
        self.dimensions = dimensions

        self.client = OpenAI(
            max_retries=0,  # We implement our own retry policy below.
            timeout=60.0,
        )

        # Recursive splitter respects paragraph and sentence boundaries
        # when possible before splitting at smaller character boundaries.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

        # A conservative interval between request start times.
        self.request_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

        # Approximate token counting for cost estimates.
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self._initialize_cache()

    # -------------------------------------------------
    # 3. SQLITE CACHE
    # -------------------------------------------------

    def _initialize_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.cache_file) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _make_cache_key(self, text):
        """
        Include the model, dimensions and exact text.
        Changing the model creates a different cache entry.
        """

        content = json.dumps(
            {
                "model": self.model,
                "dimensions": self.dimensions,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _get_cached_embedding(self, cache_key):
        with sqlite3.connect(self.cache_file) as connection:
            row = connection.execute(
                "SELECT vector_json FROM embeddings WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

        if row is None:
            return None

        return json.loads(row[0])

    def _save_cached_embeddings(self, entries):
        """
        Save completed embeddings together in one transaction.
        entries contains (cache_key, vector) pairs.
        """

        rows = [
            (key, self.model, json.dumps(vector))
            for key, vector in entries
        ]

        with sqlite3.connect(self.cache_file) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (cache_key, model, vector_json)
                VALUES (?, ?, ?)
                """,
                rows,
            )

    # -------------------------------------------------
    # 4. LOAD DOCUMENTS
    # -------------------------------------------------

    def _load_documents(self):
        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"Input directory does not exist: {self.input_dir}"
            )

        documents = []
        supported_extensions = {".pdf", ".txt", ".md"}

        files = sorted(
            path for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in supported_extensions
        )

        logger.info("Found %d supported files.", len(files))

        for file_path in files:
            relative_path = str(file_path.relative_to(self.input_dir))

            try:
                if file_path.suffix.lower() == ".pdf":
                    reader = PdfReader(str(file_path))

                    if reader.is_encrypted:
                        logger.warning("Skipping encrypted PDF: %s", relative_path)
                        continue

                    for page_number, page in enumerate(reader.pages, start=1):
                        page_text = page.extract_text() or ""

                        if page_text.strip():
                            documents.append({
                                "text": page_text,
                                "source": relative_path,
                                "page": page_number,
                            })

                else:
                    text = file_path.read_text(encoding="utf-8")

                    if text.strip():
                        documents.append({
                            "text": text,
                            "source": relative_path,
                            "page": None,
                        })

            except Exception:
                # A broken document should not stop the entire pipeline.
                logger.exception("Could not read %s; skipping it.", relative_path)

        logger.info("Loaded %d document sections.", len(documents))
        return documents

    # -------------------------------------------------
    # 5. CHUNK DOCUMENTS
    # -------------------------------------------------

    def _chunk_documents(self, documents):
        chunks = []

        for document in documents:
            split_documents = self.splitter.create_documents(
                [document["text"]]
            )

            for index, split_document in enumerate(split_documents):
                text = split_document.page_content.strip()

                if not text:
                    continue

                start_index = split_document.metadata.get("start_index", 0)

                # Stable ID for the same source, page, position and content.
                identity = (
                    f"{document['source']}|{document['page']}|"
                    f"{start_index}|{text}"
                )
                chunk_id = hashlib.sha256(
                    identity.encode("utf-8")
                ).hexdigest()

                chunks.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": document["source"],
                    "page": document["page"],
                    "chunk_index": index,
                    "start_index": start_index,
                    "cache_key": self._make_cache_key(text),
                })

        logger.info("Created %d text chunks.", len(chunks))
        return chunks

    # -------------------------------------------------
    # 6. TOKEN COUNTS AND COST ESTIMATION
    # -------------------------------------------------

    def _count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def _estimate_cost(self, token_count):
        return (
            token_count / 1_000_000
        ) * self.price_per_million_tokens

    # -------------------------------------------------
    # 7. RATE LIMITING AND RETRIES
    # -------------------------------------------------

    def _wait_for_rate_limit(self):
        elapsed = time.monotonic() - self.last_request_time
        remaining = self.request_interval - elapsed

        if remaining > 0:
            time.sleep(remaining)

        self.last_request_time = time.monotonic()

    def _embed_batch_with_retry(self, texts):
        """
        Retry common transient API errors using exponential backoff.
        Permanent errors are allowed to propagate to the caller.
        """

        for attempt in range(self.max_retries):
            self._wait_for_rate_limit()

            try:
                arguments = {
                    "model": self.model,
                    "input": texts,
                }

                if self.dimensions is not None:
                    arguments["dimensions"] = self.dimensions

                response = self.client.embeddings.create(**arguments)

                # Sort by API-provided input index to preserve alignment.
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors = [item.embedding for item in ordered]

                actual_tokens = getattr(
                    getattr(response, "usage", None),
                    "prompt_tokens",
                    None,
                )

                return vectors, actual_tokens

            except (
                RateLimitError,
                APITimeoutError,
                APIConnectionError,
                InternalServerError,
            ):
                if attempt == self.max_retries - 1:
                    logger.exception("Embedding batch failed after retries.")
                    raise

                # Exponential backoff with a small cap.
                wait_seconds = min(2 ** attempt, 30)

                logger.warning(
                    "Transient API error. Retry %d/%d in %d seconds.",
                    attempt + 1,
                    self.max_retries - 1,
                    wait_seconds,
                )

                time.sleep(wait_seconds)

        raise RuntimeError("Embedding retries exhausted.")

    # -------------------------------------------------
    # 8. EMBED IN BATCHES AND USE THE CACHE
    # -------------------------------------------------

    def _embed_chunks(self, chunks):
        total_chunks = len(chunks)
        cache_hits = 0
        cache_misses = 0
        estimated_tokens = 0
        actual_api_tokens = 0
        has_actual_token_usage = True
        api_requests = 0

        # Each vector is stored by its cache key.
        vectors_by_key = {}
        uncached_chunks = []

        logger.info("Checking embedding cache...")

        for chunk in chunks:
            vector = self._get_cached_embedding(chunk["cache_key"])

            if vector is not None:
                vectors_by_key[chunk["cache_key"]] = vector
                cache_hits += 1
            else:
                uncached_chunks.append(chunk)
                cache_misses += 1
                estimated_tokens += self._count_tokens(chunk["text"])

        logger.info(
            "Cache hits: %d | Cache misses: %d",
            cache_hits,
            cache_misses,
        )

        total_batches = (
            len(uncached_chunks) + self.batch_size - 1
        ) // self.batch_size

        for start in range(0, len(uncached_chunks), self.batch_size):
            batch = uncached_chunks[start:start + self.batch_size]
            texts = [chunk["text"] for chunk in batch]

            batch_number = start // self.batch_size + 1

            logger.info(
                "Embedding batch %d/%d (%d chunks).",
                batch_number,
                total_batches,
                len(batch),
            )

            vectors, usage_tokens = self._embed_batch_with_retry(texts)
            api_requests += 1

            if len(vectors) != len(batch):
                raise RuntimeError("Embedding count does not match batch size.")

            if usage_tokens is None:
                has_actual_token_usage = False
            else:
                actual_api_tokens += usage_tokens

            cache_entries = []

            for chunk, vector in zip(batch, vectors):
                key = chunk["cache_key"]
                vectors_by_key[key] = vector
                cache_entries.append((key, vector))

            # Save after each successful batch, so completed work survives
            # if a later batch fails.
            self._save_cached_embeddings(cache_entries)

            logger.info(
                "Completed batch %d/%d.",
                batch_number,
                total_batches,
            )

        # Build output records in the original chunk order.
        points = []

        for chunk in chunks:
            vector = vectors_by_key[chunk["cache_key"]]

            # Qdrant supports UUID strings as point IDs.
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk["chunk_id"],
            ))

            payload = {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "start_index": chunk["start_index"],
                "chunk_id": chunk["chunk_id"],
                "embedding_model": self.model,
            }

            points.append({
                "id": point_id,
                "vector": vector,
                "payload": payload,
            })

        if has_actual_token_usage:
            cost_basis = actual_api_tokens
            cost_basis_name = "API-reported tokens"
        else:
            cost_basis = estimated_tokens
            cost_basis_name = "estimated tokens for uncached chunks"

        estimated_cost = self._estimate_cost(cost_basis)

        statistics = {
            "total_chunks": total_chunks,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "api_requests": api_requests,
            "estimated_tokens_for_uncached_chunks": estimated_tokens,
            "api_reported_tokens": (
                actual_api_tokens if has_actual_token_usage else None
            ),
            "cost_basis": cost_basis_name,
            "estimated_cost_usd": round(estimated_cost, 8),
        }

        return points, statistics

    # -------------------------------------------------
    # 9. EXPORT QDRANT-READY JSONL
    # -------------------------------------------------

    def _write_qdrant_points(self, points):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temporary file first, then replace the final output.
        temporary_file = self.output_file.with_suffix(
            self.output_file.suffix + ".tmp"
        )

        with temporary_file.open("w", encoding="utf-8") as file:
            for point in points:
                file.write(json.dumps(point, ensure_ascii=False) + "\n")

        temporary_file.replace(self.output_file)

        logger.info(
            "Wrote %d Qdrant-ready points to %s.",
            len(points),
            self.output_file,
        )

    # -------------------------------------------------
    # 10. PUBLIC ENTRY POINT
    # -------------------------------------------------

    def run(self):
        start_time = time.perf_counter()

        documents = self._load_documents()
        chunks = self._chunk_documents(documents)

        if not chunks:
            logger.warning("No text chunks found. No output was generated.")
            return {
                "documents": len(documents),
                "chunks": 0,
                "output_file": None,
            }

        points, statistics = self._embed_chunks(chunks)
        self._write_qdrant_points(points)

        total_time = time.perf_counter() - start_time

        report = {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            **statistics,
            "output_file": str(self.output_file),
            "elapsed_seconds": round(total_time, 2),
        }

        logger.info("Pipeline summary:\n%s", json.dumps(report, indent=2))
        return report


# =====================================================
# 11. RUN THE PIPELINE
# =====================================================

def main():
    input_dir = input(
        "Enter the directory containing PDF/TXT/MD files: "
    ).strip().strip('"')

    output_file = input(
        "Output JSONL path [qdrant_points.jsonl]: "
    ).strip().strip('"') or "qdrant_points.jsonl"

    try:
        pipeline = EmbeddingPipeline(
            input_dir=input_dir,
            output_file=output_file,
            cache_file="embedding_cache.db",
            model="text-embedding-3-small",
            batch_size=64,
            chunk_size=1000,
            chunk_overlap=150,
            max_retries=5,
            requests_per_minute=60,
            price_per_million_tokens=0.02,
        )

        summary = pipeline.run()

        print("\n========== PIPELINE SUMMARY ==========")
        print(json.dumps(summary, indent=2))
        print("======================================")

    except Exception as error:
        logger.exception("Pipeline failed.")
        print(f"Pipeline failed: {error}")


if __name__ == "__main__":
    main()