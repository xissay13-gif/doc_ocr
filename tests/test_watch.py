"""Тесты наблюдателя за папкой (Tesseract не нужен).

Запуск:  python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from court_ocr.watch import StateStore, Watcher, file_key  # noqa: E402


def touch(path: Path, data: bytes = b"%PDF-1.4\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class WatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.inbox = self.root / "input"
        self.inbox.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def watcher(self, **kw) -> Watcher:
        kw.setdefault("stable_checks", 2)
        return Watcher([self.inbox], **kw)

    def test_file_ready_only_after_stability_window(self):
        """Файл отдаётся не сразу: сначала должен «устояться»."""
        touch(self.inbox / "a.pdf")
        w = self.watcher()
        self.assertEqual(w.poll(), [])   # первый раз только увидели
        self.assertEqual(w.poll(), [])   # второй — один интервал без изменений
        self.assertEqual([p.name for p in w.poll()], ["a.pdf"])

    def test_growing_file_is_not_taken(self):
        """Пока файл дописывается, счётчик стабильности сбрасывается."""
        path = touch(self.inbox / "big.pdf", b"x" * 10)
        w = self.watcher()
        w.poll()
        path.write_bytes(b"x" * 200)     # копирование продолжается
        w.poll()
        self.assertEqual(w.poll(), [])   # с момента изменения прошёл 1 опрос
        self.assertEqual([p.name for p in w.poll()], ["big.pdf"])

    def test_empty_file_ignored(self):
        touch(self.inbox / "zero.pdf", b"")
        w = self.watcher()
        for _ in range(4):
            self.assertEqual(w.poll(), [])

    def test_unsupported_extension_ignored(self):
        touch(self.inbox / "notes.txt")
        w = self.watcher()
        for _ in range(4):
            self.assertEqual(w.poll(), [])

    def _drain(self, w: Watcher, rounds: int = 4):
        """Прокрутить опросы и вернуть все выданные файлы, помечая их обработанными."""
        out = []
        for _ in range(rounds):
            batch = w.poll()
            for f in batch:
                w.mark_done(f)
            out.extend(batch)
        return out

    def test_processed_file_is_not_taken_twice(self):
        touch(self.inbox / "a.pdf")
        w = self.watcher()
        self.assertEqual([p.name for p in self._drain(w)], ["a.pdf"])
        self.assertEqual(self._drain(w), [])

    def test_changed_file_is_taken_again(self):
        """Тот же путь, но другое содержимое — это новый документ."""
        path = touch(self.inbox / "a.pdf", b"one")
        w = self.watcher()
        self.assertEqual(len(self._drain(w)), 1)
        path.write_bytes(b"two-and-longer")
        self.assertEqual([p.name for p in self._drain(w)], ["a.pdf"])

    def test_output_dir_inside_input_is_excluded(self):
        """Распознанный PDF в output/ не должен вернуться на вход."""
        out = self.inbox / "output"
        touch(out / "a.pdf")
        touch(self.inbox / "scan.pdf")
        w = self.watcher(exclude=[out])
        self.assertEqual([p.name for p in self._drain(w)], ["scan.pdf"])

    def test_recursive_and_flat(self):
        touch(self.inbox / "sub" / "deep.pdf")
        flat = self.watcher(recursive=False)
        self.assertEqual(self._drain(flat), [])
        deep = self.watcher(recursive=True)
        self.assertEqual([p.name for p in self._drain(deep)], ["deep.pdf"])

    def test_vanished_file_is_forgotten(self):
        path = touch(self.inbox / "a.pdf")
        w = self.watcher()
        w.poll()
        path.unlink()
        w.poll()
        self.assertEqual(w.poll(), [])

    def test_skip_existing_marks_current_files(self):
        touch(self.inbox / "old.pdf")
        w = self.watcher()
        self.assertEqual(w.skip_existing(), 1)
        touch(self.inbox / "new.pdf")
        self.assertEqual([p.name for p in self._drain(w)], ["new.pdf"])

    def test_run_stops_and_marks_done(self):
        touch(self.inbox / "a.pdf")
        w = self.watcher(interval=0.01)
        seen, idle = [], []

        def handler(files):
            seen.extend(files)

        def stop():
            return len(seen) > 0 and len(idle) > 0

        w.run(handler, on_idle=lambda: idle.append(1), stop=stop)
        self.assertEqual([p.name for p in seen], ["a.pdf"])
        self.assertIn(file_key(self.inbox / "a.pdf"), w.state)

    def test_run_marks_done_even_if_handler_fails(self):
        """Битый файл не должен крутиться в цикле вечно."""
        touch(self.inbox / "bad.pdf")
        w = self.watcher(interval=0.01)
        calls = []

        def handler(files):
            calls.append(files)
            raise RuntimeError("OCR упал")

        with self.assertRaises(RuntimeError):
            w.run(handler)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self._drain(w), [])


class StateStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_state_survives_restart(self):
        path = self.root / "out" / "state.json"
        first = StateStore(path)
        first.add(("C:/scan/a.pdf", 100, 1700000000))
        self.assertIn(("C:/scan/a.pdf", 100, 1700000000), StateStore(path))

    def test_corrupted_state_is_ignored(self):
        path = self.root / "state.json"
        path.write_text("{не json", encoding="utf-8")
        self.assertEqual(len(StateStore(path)), 0)

    def test_clear_removes_file(self):
        path = self.root / "state.json"
        store = StateStore(path)
        store.add(("a", 1, 2))
        store.clear()
        self.assertFalse(path.exists())
        self.assertEqual(len(store), 0)

    def test_memory_only_store_needs_no_path(self):
        store = StateStore()
        store.add(("a", 1, 2))
        self.assertIn(("a", 1, 2), store)


if __name__ == "__main__":
    unittest.main()
