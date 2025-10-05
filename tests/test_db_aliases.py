from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import sys
import types

telegram_stub = types.ModuleType("telegram")
telegram_stub.Message = object
telegram_stub.User = object
sys.modules.setdefault("telegram", telegram_stub)

from bot_double.db import Database


class DatabaseAliasTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        self.db = Database(Path(self._tmp.name), max_messages_per_user=100)

    def tearDown(self) -> None:
        self.db.close()
        os.unlink(self._tmp.name)

    def test_alias_reassignment_transfers_owner(self) -> None:
        chat_id = 10
        user_a = self.db.upsert_user(telegram_id=111, username="user_a", first_name="A", last_name=None)
        user_b = self.db.upsert_user(telegram_id=222, username="user_b", first_name="B", last_name=None)

        added_first, skipped_first = self.db.add_aliases(chat_id, user_a, ["даник"])
        self.assertEqual(added_first, ["даник"])
        self.assertEqual(skipped_first, [])

        added_second, skipped_second = self.db.add_aliases(chat_id, user_b, ["даник"])
        self.assertEqual(added_second, ["даник"])
        self.assertEqual(skipped_second, [])

        rows = self.db.get_aliases_for_chat(chat_id)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(int(row["user_id"]), user_b)
        self.assertEqual(row["alias"], "даник")

        added_third, skipped_third = self.db.add_aliases(chat_id, user_b, ["даник"])
        self.assertEqual(added_third, [])
        self.assertEqual(skipped_third, ["даник"])


if __name__ == "__main__":
    unittest.main()
