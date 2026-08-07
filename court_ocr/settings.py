"""Настройки программы: пути к папкам, потоки, режим мониторинга.

Приоритет источников (каждый следующий перекрывает предыдущий):

    значения по умолчанию → court-ocr.json рядом с программой →
    переменные окружения → аргументы командной строки.

Файл ``court-ocr.json`` лежит рядом с ``app.py`` (или с ``court-ocr.exe``) и
позволяет прописать любые пути::

    {
      "input_dir":  "D:/Сканы/входящие",
      "output_dir": "//server/share/ocr",
      "threads": 8,
      "watch": true,
      "watch_interval": 2.0
    }

Пути могут быть абсолютными, сетевыми (UNC), относительными (считаются от папки
программы) и содержать ``~``, ``%ПЕРЕМЕННАЯ%`` или ``$ПЕРЕМЕННАЯ``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_NAME = "court-ocr.json"

DEFAULTS: Dict[str, Any] = {
    "input_dir": "input",
    "output_dir": "output",
    # Куда класть CSV и распознанные PDF по отдельности. Пусто — оба в output_dir.
    "csv_dir": "",
    "pdf_dir": "",
    "threads": 0,            # 0 — по числу ядер CPU
    "recursive": True,       # заходить в подпапки входной папки
    "watch": True,           # непрерывный мониторинг вместо разового прохода
    "watch_interval": 2.0,   # период опроса папки, с
    "watch_stable": 2,       # опросов подряд без изменений — файл дописан
    "make_pdf": True,        # делать распознанный PDF рядом с CSV
}

# Переменная окружения → ключ настроек.
ENV_MAP = {
    "COURT_OCR_INPUT": "input_dir",
    "COURT_OCR_OUTPUT": "output_dir",
    "COURT_OCR_CSV_DIR": "csv_dir",
    "COURT_OCR_PDF_DIR": "pdf_dir",
    "COURT_OCR_THREADS": "threads",
    "COURT_OCR_RECURSIVE": "recursive",
    "COURT_OCR_WATCH": "watch",
    "COURT_OCR_INTERVAL": "watch_interval",
}

_TRUE = {"1", "true", "yes", "on", "да", "y", "д"}
_FALSE = {"0", "false", "no", "off", "нет", "n", "н"}


def _coerce(key: str, value: Any) -> Any:
    """Привести значение к типу значения по умолчанию (в env всё — строки)."""
    default = DEFAULTS[key]
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        return default
    if isinstance(default, int):
        try:
            return int(str(value).strip())
        except ValueError:
            return default
    if isinstance(default, float):
        try:
            return float(str(value).strip().replace(",", "."))
        except ValueError:
            return default
    return str(value)


def config_path(base: Path) -> Path:
    return Path(base) / CONFIG_NAME


def load(base: Path) -> Dict[str, Any]:
    """Собрать настройки: умолчания → court-ocr.json → переменные окружения."""
    data = dict(DEFAULTS)

    path = config_path(base)
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            sys.stderr.write(f"[!] Не удалось прочитать {path}: {exc}\n"
                             f"    Используются настройки по умолчанию.\n")
            raw = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if key in DEFAULTS:
                    data[key] = _coerce(key, value)

    for env, key in ENV_MAP.items():
        value = os.environ.get(env)
        if value not in (None, ""):
            data[key] = _coerce(key, value)

    return data


def save(base: Path, data: Dict[str, Any]) -> Path:
    """Записать настройки в court-ocr.json рядом с программой."""
    path = config_path(base)
    payload = {key: data.get(key, DEFAULTS[key]) for key in DEFAULTS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def resolve_optional_dir(base: Path, value: Any) -> Optional[Path]:
    """Путь, который можно не задавать: пусто → None.

    Раскрывает ~, %ПЕРЕМЕННУЮ%/$ПЕРЕМЕННУЮ и снимает кавычки (путь часто
    копируют из Проводника вместе с ними). Относительный путь считается от
    папки программы."""
    text = os.path.expandvars(str(value or "")).strip().strip('"')
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else Path(base) / path


def resolve_dir(base: Path, value: Any, default: str = "output") -> Path:
    """То же, но для обязательного пути: пусто → значение default."""
    return resolve_optional_dir(base, value) or resolve_optional_dir(base, default) \
        or Path(base) / default
