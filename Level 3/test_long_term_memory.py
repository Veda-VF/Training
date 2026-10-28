# python -m unittest -v test_long_term_memory.py

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from long_term_memory_agent import LongTermMemory


class TestLongTermMemory(unittest.TestCase):

    def setUp(self):
        # Give each test its own temporary JSON file.
        self.temp_dir = tempfile.TemporaryDirectory()
        filepath = os.path.join(
            self.temp_dir.name,
            "test_memory.json",
        )
        self.memory = LongTermMemory(filepath=filepath)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cleanup_expired(self):
        """Expired memories are removed; unexpired ones remain."""
        now = datetime.now()

        expired = self.memory.store(
            "Old preference",
            ttl=7,
        )
        expired.timestamp = (now - timedelta(days=8)).isoformat()

        active = self.memory.store(
            "Current project",
            ttl=30,
        )
        active.timestamp = (now - timedelta(days=8)).isoformat()

        removed = self.memory.cleanup_expired(now=now)

        self.assertEqual(removed, 1)
        self.assertEqual(len(self.memory.entries), 1)
        self.assertEqual(self.memory.entries[0].id, active.id)

    def test_decay_factor(self):
        """Check decay for memories aged 10 and 100 days."""
        now = datetime.now()

        ten_day_entry = self.memory.store("Memory from 10 days ago")
        ten_day_entry.timestamp = (
            now - timedelta(days=10)
        ).isoformat()

        hundred_day_entry = self.memory.store("Memory from 100 days ago")
        hundred_day_entry.timestamp = (
            now - timedelta(days=100)
        ).isoformat()

        factor_10 = self.memory.decay_factor(
            ten_day_entry, now=now
        )
        factor_100 = self.memory.decay_factor(
            hundred_day_entry, now=now
        )

        self.assertAlmostEqual(factor_10, 0.909, places=3)
        self.assertAlmostEqual(factor_100, 0.5, places=3)
        self.assertGreater(factor_10, factor_100)

    def test_deduplicate_exact_duplicates(self):
        """Two identical facts should become one memory."""
        self.memory.store(
            "User's name is Alice",
            category="identity",
            user_id="alice",
        )
        self.memory.store(
            "User's name is Alice",
            category="identity",
            user_id="alice",
        )

        removed = self.memory.deduplicate(threshold=0.95)

        self.assertEqual(removed, 1)
        self.assertEqual(len(self.memory.entries), 1)

    def test_deduplicate_keeps_detail_and_earliest_timestamp(self):
        """Merging preserves longer content and the older timestamp."""
        now = datetime.now()

        short = self.memory.store(
            "User prefers Python",
            category="preference",
            user_id="alice",
        )
        short.timestamp = (now - timedelta(days=10)).isoformat()

        detailed = self.memory.store(
            "User prefers Python for data science projects",
            category="preference",
            user_id="alice",
        )
        detailed.timestamp = (now - timedelta(days=2)).isoformat()

        # Control similarity so this test focuses on merge behavior.
        with patch("long_term_memory_agent.cosine_sim", return_value=0.99):
            removed = self.memory.deduplicate(threshold=0.95)

        self.assertEqual(removed, 1)
        self.assertEqual(len(self.memory.entries), 1)

        merged = self.memory.entries[0]
        self.assertEqual(
            merged.content,
            "User prefers Python for data science projects",
        )
        self.assertEqual(
            merged.timestamp,
            (now - timedelta(days=10)).isoformat(),
        )

    def test_deduplicate_preserves_user_isolation(self):
        """Similar memories from different users must not merge."""
        self.memory.store(
            "User's name is Alice",
            category="identity",
            user_id="alice",
        )
        self.memory.store(
            "User's name is Alice",
            category="identity",
            user_id="bob",
        )

        removed = self.memory.deduplicate(threshold=0.95)

        self.assertEqual(removed, 0)
        self.assertEqual(len(self.memory.entries), 2)

    def test_invalid_ttl(self):
        """A zero or negative TTL is rejected."""
        with self.assertRaises(ValueError):
            self.memory.store("Invalid memory", ttl=0)

        with self.assertRaises(ValueError):
            self.memory.store("Invalid memory", ttl=-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)