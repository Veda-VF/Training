"""
A self-contained long-term memory system for AI agents.

Feature	                        Behavior
-------------------------   ----------------------------------------
Default TTL	                365 days
Expiration	                Removes a memory when its age is greater than its TTL
Decay at 10 days	        Approximately 0.91
Decay at 100 days	        0.50
Deduplication threshold	    0.95 by default
Merged content	            Longer content is retained
Merged timestamp		    Earliest timestamp is retained
User isolation	            Memories from different users are never merged
Persistence	                Changes are saved to JSON
"""

import json
import os
import re
import hashlib
from math import sqrt
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Embedding & similarity
# ---------------------------------------------------------------------------

def simple_embed(text: str, dim: int = 128) -> list[float]:
    """Deterministic pseudo-embedding based on word hashing."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return [0.0] * dim
    vector = [0.0] * dim
    for word in words:
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        for j in range(dim):
            idx = (h + j * 7) % 10000
            vector[j] += ((idx % 200) - 100) / 100.0
    magnitude = sqrt(sum(v * v for v in vector))
    if magnitude > 0:
        vector = [v / magnitude for v in vector]
    return vector


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sqrt(sum(x * x for x in a)) or 1e-9
    mag_b = sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

FACT_PATTERNS = [
    (r"my name is (\w+)", "identity"),
    (r"I (?:like|love|prefer|enjoy) (.+?)(?:[\.,!]|$)", "preference"),
    (r"I (?:hate|dislike|don't like|cannot stand) (.+?)(?:[\.,!]|$)", "dislike"),
    (r"I(?:'m| am) (?:allergic|allergies) (?:to )?(.+?)(?:[\.,!]|$)", "health"),
    (r"I work (?:at|for) (.+?)(?:[\.,!]|$)", "work"),
    (r"I live in (.+?)(?:[\.,!]|$)", "location"),
    (r"my (?:favorite|favourite) (\w+) is (.+?)(?:[\.,!]|$)", "preference"),
    (r"I have (?:a |an )?(.+?)(?:[\.,!]|$)", "possession"),
    (r"I'm (?:studying|learning) (.+?)(?:[\.,!]|$)", "education"),
    (r"my (?:dog|cat|pet) is called (\w+)", "pet"),
]


def extract_facts(message: str) -> list[dict]:
    """Extract structured facts from a user message using patterns."""
    found = []
    for pattern, category in FACT_PATTERNS:
        matches = re.findall(pattern, message, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                content = f"User's {match[0]} is {match[1]}"
            else:
                content = f"User mentioned {match}"
            found.append({"content": content, "category": category})
    return found


# ---------------------------------------------------------------------------
# LongTermMemory
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    id: str
    content: str
    category: str
    timestamp: str
    user_id: str
    source_conversation: str
    embedding: list[float] = field(default_factory=list)
    ttl: float = 365.0  # Default lifetime: 365 days

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "source_conversation": self.source_conversation,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            id=data["id"],
            content=data["content"],
            category=data["category"],
            timestamp=data["timestamp"],
            user_id=data["user_id"],
            source_conversation=data.get("source_conversation", "unknown"),
            embedding=[],  # Recomputed when loading
            ttl=float(data.get("ttl", 365.0)),
        )


class LongTermMemory:
    """
    Persistent long-term memory with semantic retrieval.

    Stores facts as vector entries, supports top-k retrieval by
    cosine similarity, and persists everything to a JSON file.
    """

    def __init__(self, filepath: str = "long_term_memory.json", embed_dim: int = 128):
        self.filepath = filepath
        self.embed_dim = embed_dim
        self.entries: list[MemoryEntry] = []
        self._next_id: int = 1
        self._load()

    # -- Public API --------------------------------------------------------

    def store(
        self,
        content: str,
        category: str = "general",
        user_id: str = "default",
        source_conversation: str = "unknown",
        ttl: float = 365.0,
    ) -> MemoryEntry:
        """Store a fact with a time-to-live in days."""
        if ttl <= 0:
            raise ValueError("TTL must be greater than 0 days.")

        entry = MemoryEntry(
            id=str(self._next_id),
            content=content,
            category=category,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            source_conversation=source_conversation,
            embedding=simple_embed(content, self.embed_dim),
            ttl=ttl,
        )

        self.entries.append(entry)
        self._next_id += 1
        self._save()
        return entry

    def store_from_message(
        self,
        message: str,
        user_id: str = "default",
        source_conversation: str = "unknown",
    ) -> list[MemoryEntry]:
        """Extract facts from a message and store them all."""
        facts = extract_facts(message)
        stored = []
        for fact in facts:
            entry = self.store(
                content=fact["content"],
                category=fact["category"],
                user_id=user_id,
                source_conversation=source_conversation,
            )
            stored.append(entry)
        return stored

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve the top-k most relevant memories for a query."""
        query_embedding = simple_embed(query, self.embed_dim)

        candidates = self.entries
        if user_id is not None:
            candidates = [e for e in candidates if e.user_id == user_id]
        if category is not None:
            candidates = [e for e in candidates if e.category == category]

        scored = []
        for entry in candidates:
            # Calculate semantic similarity first
            similarity = cosine_sim(query_embedding, entry.embedding)

            # Reduce the score as the memory gets older
            factor = self.decay_factor(entry)
            adjusted_score = similarity * factor

            if adjusted_score >= min_similarity:
                scored.append((entry, adjusted_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_context_for_query(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> str:
        """Return a formatted context string of relevant memories."""
        results = self.retrieve(query, top_k=top_k, user_id=user_id)
        if not results:
            return ""
        lines = ["[Relevant long-term memories]"]
        for entry, score in results:
            lines.append(f"  - {entry.content} (relevance: {score:.3f}, stored: {entry.timestamp})")
        return "\n".join(lines)

    def clear(self):
        self.entries = []
        self._next_id = 1
        self._save()

    def stats(self) -> dict:
        categories = {}
        for e in self.entries:
            categories[e.category] = categories.get(e.category, 0) + 1
        return {
            "total_entries": len(self.entries),
            "categories": categories,
            "filepath": self.filepath,
        }

    # -- Persistence -------------------------------------------------------

    def _save(self):
        data = {
            "next_id": self._next_id,
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self):
        if not os.path.exists(self.filepath):
            return
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._next_id = data.get("next_id", 1)
        for entry_data in data.get("entries", []):
            entry = MemoryEntry.from_dict(entry_data)
            entry.embedding = simple_embed(entry.content, self.embed_dim)
            self.entries.append(entry)

    def cleanup_expired(self, now: Optional[datetime] = None) -> int:
        """Remove memories older than their TTL and return the removal count."""
        now = now or datetime.now()
        remaining = []

        for entry in self.entries:
            stored_at = datetime.fromisoformat(entry.timestamp)
            age_days = (now - stored_at).total_seconds() / 86400

            if age_days > entry.ttl:
                continue  # Expired: do not retain this entry

            remaining.append(entry)

        removed_count = len(self.entries) - len(remaining)

        if removed_count:
            self.entries = remaining
            self._save()

        return removed_count

    def decay_factor(
        self,
        entry: MemoryEntry,
        now: Optional[datetime] = None,
    ) -> float:
        """Calculate an age-based multiplier for a memory's similarity score."""
        now = now or datetime.now()
        stored_at = datetime.fromisoformat(entry.timestamp)

        age_days = max(
            0.0,
            (now - stored_at).total_seconds() / 86400,
        )

        return 1.0 / (1.0 + age_days / 100.0)

    def deduplicate(self, threshold: float = 0.95) -> int:
        """
        Merge highly similar memories belonging to the same user and category.
        Keeps the longer content and earliest timestamp.
        Returns the number of entries removed.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        # Process older entries first, making the earliest timestamp
        # the one retained when duplicates are merged.
        ordered = sorted(
            self.entries,
            key=lambda entry: datetime.fromisoformat(entry.timestamp),
        )

        unique_entries = []

        for entry in ordered:
            duplicate = None

            for existing in unique_entries:
                # Never merge memories belonging to different users
                # or different categories.
                if (
                    entry.user_id != existing.user_id
                    or entry.category != existing.category
                ):
                    continue

                similarity = cosine_sim(
                    entry.embedding,
                    existing.embedding,
                )

                if similarity > threshold:
                    duplicate = existing
                    break

            if duplicate is None:
                unique_entries.append(entry)
                continue

            # Keep the more detailed content.
            if len(entry.content) > len(duplicate.content):
                duplicate.content = entry.content

            # Preserve the earliest timestamp.
            duplicate.timestamp = min(
                duplicate.timestamp,
                entry.timestamp,
                key=datetime.fromisoformat,
            )

            # Preserve the longer TTL so merging does not unexpectedly
            # shorten the memory's configured lifetime.
            duplicate.ttl = max(duplicate.ttl, entry.ttl)

            # Refresh the embedding to match the final merged content.
            duplicate.embedding = simple_embed(
                duplicate.content,
                self.embed_dim,
            )

        removed_count = len(self.entries) - len(unique_entries)

        if removed_count:
            self.entries = unique_entries
            self._save()

        return removed_count

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("LongTermMemory Demo")
    print("=" * 60)

    # Use a temp file for demo
    filepath = "demo_long_term_memory.json"
    if os.path.exists(filepath):
        os.remove(filepath)

    ltm = LongTermMemory(filepath=filepath)

    # Simulate a multi-session conversation
    print("\n--- Session 1: User Alice ---")
    messages = [
        "Hi, my name is Alice and I live in Berlin.",
        "I work at TechCorp as a software engineer.",
        "I love hiking and photography.",
        "I'm allergic to shellfish.",
    ]
    for msg in messages:
        stored = ltm.store_from_message(msg, user_id="alice", source_conversation="conv_1")
        for s in stored:
            print(f"  Stored: {s.content}")

    print(f"\n--- Session 2: User Bob ---")
    bob_messages = [
        "Hello, my name is Bob.",
        "I live in Tokyo and I'm learning machine learning.",
        "I prefer dark mode in all my editors.",
    ]
    for msg in bob_messages:
        stored = ltm.store_from_message(msg, user_id="bob", source_conversation="conv_2")
        for s in stored:
            print(f"  Stored: {s.content}")

    # Retrieve memories for Alice
    print("\n--- Retrieval for Alice: 'What food should I avoid?' ---")
    context = ltm.get_context_for_query("What food should I avoid?", user_id="alice")
    print(context)

    # Retrieve memories for Bob
    print("\n--- Retrieval for Bob: 'What is Bob studying?' ---")
    context = ltm.get_context_for_query("studying learning education", user_id="bob")
    print(context)

    # Cross-user search
    print("\n--- Cross-user search: 'Where do people live?' ---")
    results = ltm.retrieve("location city where live", top_k=3)
    for entry, score in results:
        print(f"  [{entry.user_id}] {entry.content} (score={score:.3f})")

    # Stats
    print(f"\n--- Memory Stats ---")
    print(json.dumps(ltm.stats(), indent=2))

    # Simulate restart: create new instance loading from same file
    print("\n--- Simulating restart (loading from disk) ---")
    ltm2 = LongTermMemory(filepath=filepath)
    print(f"Reloaded {len(ltm2.entries)} entries")
    context = ltm2.get_context_for_query("allergy health", user_id="alice", top_k=2)
    print(context)

    # Cleanup
    os.remove(filepath)
    print("\nDemo complete.")


if __name__ == "__main__":
    main()
