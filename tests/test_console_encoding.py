"""Русский вывод не должен падать, когда stdout не UTF-8.

Живой случай: в Windows у перенаправленного в файл или конвейер stdout кодировка
берётся из локали (cp1252 и подобные), и первый же русский print валится с
UnicodeEncodeError. Так сборка на GitHub Actions и упала на setup_tesseract.py.

Точек входа три, забыть перенастройку консоли легко в любой — проверяем все.
PYTHONIOENCODING=cp1252 воспроизводит ту же кодировку на любой ОС.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY_POINTS = ["setup_tesseract.py", "app.py", "ocr_court.py"]


class ConsoleEncodingTest(unittest.TestCase):
    def _run_help(self, script: str, encoding: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, PYTHONIOENCODING=encoding)
        env.pop("PYTHONUTF8", None)  # иначе Python сам всё починит и проверка ослепнет
        return subprocess.run([sys.executable, script, "--help"],
                              cwd=ROOT, env=env, capture_output=True, timeout=120)

    def test_help_survives_non_utf8_stdout(self):
        for script in ENTRY_POINTS:
            with self.subTest(script=script):
                proc = self._run_help(script, "cp1252")
                self.assertEqual(
                    proc.returncode, 0,
                    f"{script} упал при cp1252-выводе:\n"
                    f"{proc.stderr.decode('utf-8', 'replace')}")

    def test_help_still_works_in_utf8(self):
        for script in ENTRY_POINTS:
            with self.subTest(script=script):
                proc = self._run_help(script, "utf-8")
                self.assertEqual(proc.returncode, 0)
                self.assertIn("usage", proc.stdout.decode("utf-8", "replace").lower())


if __name__ == "__main__":
    unittest.main()
