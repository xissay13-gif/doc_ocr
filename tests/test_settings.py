"""Тесты настроек: court-ocr.json, переменные окружения, разбор путей."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from court_ocr import settings  # noqa: E402


class SettingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self._env = dict(os.environ)
        for key in settings.ENV_MAP:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env)
        self._tmp.cleanup()

    def write_config(self, data: dict) -> None:
        settings.config_path(self.base).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_defaults_without_config(self):
        self.assertEqual(settings.load(self.base), settings.DEFAULTS)

    def test_config_overrides_defaults(self):
        self.write_config({"input_dir": "D:/Сканы", "threads": 8})
        data = settings.load(self.base)
        self.assertEqual(data["input_dir"], "D:/Сканы")
        self.assertEqual(data["threads"], 8)
        self.assertEqual(data["output_dir"], settings.DEFAULTS["output_dir"])

    def test_env_overrides_config(self):
        self.write_config({"input_dir": "from-file"})
        os.environ["COURT_OCR_INPUT"] = "from-env"
        self.assertEqual(settings.load(self.base)["input_dir"], "from-env")

    def test_env_types_are_coerced(self):
        os.environ["COURT_OCR_THREADS"] = "12"
        os.environ["COURT_OCR_WATCH"] = "нет"
        os.environ["COURT_OCR_INTERVAL"] = "1,5"
        data = settings.load(self.base)
        self.assertEqual(data["threads"], 12)
        self.assertIs(data["watch"], False)
        self.assertEqual(data["watch_interval"], 1.5)

    def test_unknown_keys_ignored(self):
        self.write_config({"input_dir": "in", "чтотоещё": 1})
        self.assertNotIn("чтотоещё", settings.load(self.base))

    def test_broken_config_falls_back_to_defaults(self):
        settings.config_path(self.base).write_text("{сломано", encoding="utf-8")
        self.assertEqual(settings.load(self.base)["input_dir"],
                         settings.DEFAULTS["input_dir"])

    def test_save_then_load_roundtrip(self):
        data = dict(settings.DEFAULTS, input_dir="D:/Сканы", threads=4, watch=False)
        settings.save(self.base, data)
        self.assertEqual(settings.load(self.base), data)

    def test_relative_path_is_resolved_against_base(self):
        self.assertEqual(settings.resolve_dir(self.base, "input"), self.base / "input")

    def test_absolute_path_kept(self):
        absolute = self.base / "elsewhere"
        self.assertEqual(settings.resolve_dir(self.base, str(absolute)), absolute)

    def test_env_var_and_tilde_expanded(self):
        os.environ["SCAN_ROOT"] = str(self.base / "сканы")
        self.assertEqual(settings.resolve_dir(self.base, "$SCAN_ROOT/входящие"),
                         self.base / "сканы" / "входящие")
        self.assertTrue(settings.resolve_dir(self.base, "~/ocr").is_absolute())

    def test_quotes_and_spaces_stripped(self):
        """Путь, скопированный из Проводника, часто приходит в кавычках."""
        absolute = self.base / "papka"
        self.assertEqual(settings.resolve_dir(self.base, f'  "{absolute}" '), absolute)


if __name__ == "__main__":
    unittest.main()
