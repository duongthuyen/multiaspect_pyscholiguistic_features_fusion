"""Tests for scripts/utils/logging_utils.py — setup_logging."""

import logging
import tempfile
import unittest
from pathlib import Path

from scripts.utils.logging_utils import setup_logging


class SetupLoggingTests(unittest.TestCase):
    def _reset_logging(self):
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()

    def setUp(self):
        # Remove all handlers from the root logger before each test so tests
        # are independent of each other and of module-level side-effects.
        self._reset_logging()

    def tearDown(self):
        # Clean up any handlers added by the test
        self._reset_logging()

    def test_adds_stream_handler(self):
        setup_logging()
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        self.assertGreaterEqual(len(stream_handlers), 1)

    def test_adds_file_handler_when_log_file_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "test.log"
            setup_logging(log_file=log_path)
            root = logging.getLogger()
            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            self.assertEqual(len(file_handlers), 1)
            self._reset_logging()

    def test_creates_log_file_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "subdir" / "deep" / "run.log"
            setup_logging(log_file=log_path)
            self.assertTrue(log_path.parent.exists())
            self._reset_logging()

    def test_log_written_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "out.log"
            setup_logging(log_file=log_path)
            logging.getLogger("test_logger").info("hello from test")
            # Flush all handlers
            for h in logging.getLogger().handlers:
                h.flush()
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("hello from test", content)
            self._reset_logging()

    def test_double_call_does_not_duplicate_stream_handler(self):
        setup_logging()
        setup_logging()
        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(len(stream_handlers), 1)

    def test_double_call_does_not_duplicate_file_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "out.log"
            setup_logging(log_file=log_path)
            setup_logging(log_file=log_path)
            root = logging.getLogger()
            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            self.assertEqual(len(file_handlers), 1)
            self._reset_logging()

    def test_sets_root_level_to_info_by_default(self):
        setup_logging()
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_custom_level_respected(self):
        setup_logging(level=logging.WARNING)
        self.assertEqual(logging.getLogger().level, logging.WARNING)

    def test_no_file_handler_when_log_file_is_none(self):
        setup_logging(log_file=None)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(file_handlers), 0)


if __name__ == "__main__":
    unittest.main()
