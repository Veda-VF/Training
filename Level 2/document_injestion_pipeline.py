# Document Ingestion Pipeline
"""
We'll build a Python pipeline that reads .txt and .md files, splits them into chunks of 500 characters with 50 characters of overlap, 
generates embeddings, and stores them in a Qdrant vector database. Each chunk will include metadata and a deterministic ID created 
from the file path + chunk index. Running the pipeline again will reuse those IDs, so Qdrant updates existing points instead of creating duplicate points.

Install the required packages:
 python -m pip install openai langchain-text-splitters qdrant-client

Set the API key for your current PowerShell session:
$env:OPENAI_API_KEY = "your-api-key"

For local Qdrant, start a Qdrant server separately. Its default URL is http://localhost:6333.
"""

import os
import hashlib
import logging
import uuid
from pathlib import Path

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536
BATCH_SIZE = 32

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"

# A fixed namespace makes generated UUIDs deterministic.
ID_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


# --------------------------------------------------
# 2. LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. DOCUMENT INGESTION PIPELINE
# --------------------------------------------------

class DocumentIngestionPipeline:

    def __init__(
        self,
        directory,
        collection_name=COLLECTION_NAME,
        qdrant_url=QDRANT_URL,
        batch_size=BATCH_SIZE
    ):
        self.directory = Path(directory).expanduser().resolve()
        self.collection_name = collection_name
        self.batch_size = batch_size

        if not self.directory.is_dir():
            raise NotADirectoryError(
                f"Directory does not exist: {self.directory}"
            )

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "Set the OPENAI_API_KEY environment variable first."
            )

        self.openai = OpenAI()
        self.qdrant = QdrantClient(url=qdrant_url)

        # Split documents into 500-character chunks
        # with 50 characters of overlap.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True
        )

        self._create_collection()

    # --------------------------------------------------
    # 4. CREATE QDRANT COLLECTION
    # --------------------------------------------------

    def _create_collection(self):
        if not self.qdrant.collection_exists(
            collection_name=self.collection_name
        ):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(
                "Created Qdrant collection: %s",
                self.collection_name
            )
        else:
            logger.info(
                "Using existing collection: %s",
                self.collection_name
            )

    # --------------------------------------------------
    # 5. FIND DOCUMENTS
    # --------------------------------------------------

    def find_documents(self):
        supported_extensions = {".txt", ".md"}

        files = [
            path
            for path in self.directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in supported_extensions
        ]

        files.sort()

        logger.info("Found %d documents.", len(files))
        return files

    # --------------------------------------------------
    # 6. READ AND CHUNK A DOCUMENT
    # --------------------------------------------------

    def read_and_chunk(self, file_path):
        try:
            content = file_path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            logger.error(
                "Could not read %s: %s",
                file_path,
                error
            )
            return []

        if not content.strip():
            logger.warning("Skipping empty file: %s", file_path)
            return []

        chunks = self.splitter.create_documents(
            [content],
            metadatas=[{"source_file": str(
                file_path.relative_to(self.directory)
            )}]
        )

        documents = []

        for index, chunk in enumerate(chunks):
            documents.append({
                "text": chunk.page_content,
                "source_file": str(
                    file_path.relative_to(self.directory)
                ),
                "chunk_index": index,
                "file_type": file_path.suffix.lower(),
                "start_index": chunk.metadata.get("start_index", 0)
            })

        logger.info(
            "Chunked %s into %d chunks.",
            file_path.name,
            len(documents)
        )

        return documents

    # --------------------------------------------------
    # 7. GENERATE A DETERMINISTIC POINT ID
    # --------------------------------------------------

    @staticmethod
    def make_point_id(source_file, chunk_index):
        """
        Same relative file path + chunk index
        produces the same UUID every time.
        """

        unique_key = f"{source_file}:{chunk_index}"

        return str(
            uuid.uuid5(ID_NAMESPACE, unique_key)
        )

    # --------------------------------------------------
    # 8. GENERATE EMBEDDINGS AND INSERT INTO QDRANT
    # --------------------------------------------------

    def embed_and_upsert(self, documents):
        if not documents:
            return 0

        inserted_count = 0

        # Process chunks in manageable batches.
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start:start + self.batch_size]
            texts = [item["text"] for item in batch]

            try:
                response = self.openai.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=texts
                )
            except Exception:
                # Stop rather than silently losing a failed batch.
                logger.exception(
                    "Embedding request failed for batch starting at %d.",
                    start
                )
                raise

            points = []

            for item, embedding_result in zip(
                batch, response.data
            ):
                point_id = self.make_point_id(
                    item["source_file"],
                    item["chunk_index"]
                )

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding_result.embedding,
                        payload={
                            "text": item["text"],
                            "source_file": item["source_file"],
                            "chunk_index": item["chunk_index"],
                            "file_type": item["file_type"],
                            "start_index": item["start_index"],
                            "embedding_model": EMBEDDING_MODEL
                        }
                    )
                )

            # Upsert inserts new points or updates existing IDs.
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

            inserted_count += len(points)

            logger.info(
                "Upserted %d/%d chunks.",
                min(start + len(batch), len(documents)),
                len(documents)
            )

        return inserted_count

    # --------------------------------------------------
    # 9. RUN THE COMPLETE PIPELINE
    # --------------------------------------------------

    def run(self):
        files = self.find_documents()

        if not files:
            logger.warning("No .txt or .md files found.")
            return 0

        all_documents = []

        for file_path in files:
            documents = self.read_and_chunk(file_path)
            all_documents.extend(documents)

        logger.info(
            "Total chunks ready for embedding: %d",
            len(all_documents)
        )

        total_upserted = self.embed_and_upsert(all_documents)

        logger.info(
            "Ingestion complete. Processed %d chunks.",
            total_upserted
        )

        return total_upserted


# --------------------------------------------------
# 10. MAIN PROGRAM
# --------------------------------------------------

def main():
    print("===== DOCUMENT INGESTION PIPELINE =====")

    directory = input(
        "Enter the directory containing .txt and .md files: "
    ).strip().strip('"')

    try:
        pipeline = DocumentIngestionPipeline(directory)
        count = pipeline.run()

        print(f"\nPipeline completed. Chunks upserted: {count}")
        print(f"Qdrant collection: {COLLECTION_NAME}")

    except Exception as error:
        logger.exception("Pipeline failed.")
        print(f"\nPipeline failed: {error}")


if __name__ == "__main__":
    main()