"""Обёртка над встроенным Tesseract: поиск бинарника, OSD и распознавание (TSV).

Tesseract вызывается как внешний процесс (tesseract.exe), поэтому один OCR-проход
не блокирует GIL — это позволяет обрабатывать страницы в несколько потоков.

За один проход мы получаем TSV: и текст, и уверенность по каждому слову. Отдельный
текстовый проход не нужен — текст собирается из слов TSV.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

# Не показывать мелькающие консольные окна при вызове tesseract в потоках (Windows).
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class TesseractNotFound(RuntimeError):
    pass


def find_tesseract(explicit: Optional[str] = None) -> Optional[str]:
    """Ищем tesseract.exe: явный путь → переменная окружения → встроенный → PATH."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("TESSERACT_CMD")
    if env:
        candidates.append(env)
    exe = "tesseract.exe" if sys.platform == "win32" else "tesseract"
    # Встроенный: во временной распаковке onefile (_MEIPASS), рядом с .exe
    # (onedir) и рядом с корнем проекта.
    roots = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            roots.append(Path(meipass))
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parent.parent)
    roots.append(Path.cwd())
    for root in roots:
        candidates.append(str(root / "tesseract" / exe))
    for c in candidates:
        if c and Path(c).is_file():
            return str(Path(c).resolve())
    found = shutil.which("tesseract")
    return found


def find_tessdata(tess_cmd: str) -> Optional[str]:
    """Папка tessdata рядом с бинарником (встроенные языковые модели)."""
    d = Path(tess_cmd).resolve().parent / "tessdata"
    if d.is_dir():
        return str(d)
    return None


@dataclass
class OSDResult:
    orientation: Optional[int]  # 0 / 90 / 180 / 270 — как «стоит» страница
    rotate: Optional[int]       # на сколько градусов по часовой довернуть до «прямо»
    confidence: Optional[float]


@dataclass
class OCRResult:
    text: str
    confidence: float           # средняя по словам, 0..100 (−1-строки отброшены)
    n_words: int
    lines: List[str] = field(default_factory=list)  # распознанные строки по порядку


class Tesseract:
    """Конфигурированный вызыватель Tesseract."""

    def __init__(
        self,
        cmd: str,
        tessdata: Optional[str] = None,
        lang: str = "rus",
        psm: int = 6,
        oem: int = 1,
        dpi: int = 300,
    ):
        self.cmd = cmd
        self.tessdata = tessdata
        self.lang = lang
        self.psm = psm
        self.oem = oem
        self.dpi = dpi

    # -- внутреннее ------------------------------------------------------- #
    def _base_args(self) -> list[str]:
        args = [self.cmd]
        if self.tessdata:
            args += ["--tessdata-dir", self.tessdata]
        return args

    def _run(self, image: Image.Image, tail: list[str], timeout: int = 120) -> str:
        """Сохраняем картинку во временный PNG и запускаем tesseract → stdout+stderr."""
        fd, path = tempfile.mkstemp(suffix=".png", prefix="ocr_")
        os.close(fd)
        try:
            image.save(path, format="PNG")
            proc = subprocess.run(
                self._base_args() + [path, "stdout"] + tail,
                capture_output=True,
                timeout=timeout,
                creationflags=_CREATE_NO_WINDOW,
            )
            out = proc.stdout.decode("utf-8", "replace")
            err = proc.stderr.decode("utf-8", "replace")
            return out + "\n" + err
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # -- OSD (ориентация) ------------------------------------------------- #
    def osd(self, image: Image.Image) -> Optional[OSDResult]:
        """Определение ориентации 0/90/180/270. None — если OSD не отработал."""
        try:
            text = self._run(image, ["--psm", "0"], timeout=60)
        except (subprocess.TimeoutExpired, OSError):
            return None

        def grab(key: str, cast):
            m = re.search(rf"{key}\s*:\s*([-\d.]+)", text)
            return cast(m.group(1)) if m else None

        orient = grab("Orientation in degrees", lambda s: int(float(s)))
        rotate = grab("Rotate", lambda s: int(float(s)))
        conf = grab("Orientation confidence", float)
        if rotate is None and orient is None:
            return None
        return OSDResult(orientation=orient, rotate=rotate, confidence=conf)

    # -- Распознавание + searchable-PDF за один проход -------------------- #
    def image_to_document(self, image: Image.Image):
        """OCR: вернуть (OCRResult, pdf_bytes). PDF — картинка с невидимым
        текстовым слоем (можно искать/копировать текст). Данные и PDF получаются
        за один проход tesseract."""
        fd, png = tempfile.mkstemp(suffix=".png", prefix="ocr_")
        os.close(fd)
        base = png[:-4]                       # tesseract допишет .tsv / .pdf
        tsv_path, pdf_path = base + ".tsv", base + ".pdf"
        try:
            image.save(png, format="PNG")
            tail = [
                "-l", self.lang, "--oem", str(self.oem), "--psm", str(self.psm),
                "--dpi", str(self.dpi),
                "-c", "preserve_interword_spaces=1",
                "-c", "tessedit_create_tsv=1",
                "-c", "tessedit_create_pdf=1",
            ]
            subprocess.run(
                self._base_args() + [png, base] + tail,
                capture_output=True, timeout=240, creationflags=_CREATE_NO_WINDOW,
            )
            tsv_raw = ""
            if os.path.exists(tsv_path):
                with open(tsv_path, encoding="utf-8", errors="replace") as f:
                    tsv_raw = f.read()
            pdf_bytes = b""
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
            return _parse_tsv(tsv_raw), pdf_bytes
        finally:
            for p in (png, tsv_path, pdf_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # -- Распознавание (TSV → текст + уверенность) ------------------------ #
    def image_to_data(self, image: Image.Image) -> OCRResult:
        # Используем переменную tessedit_create_tsv=1 вместо конфиг-файла «tsv»,
        # чтобы не зависеть от наличия папки tessdata/configs (её нет, например, в
        # отдельно скачанной tessdata_best) — иначе tesseract тихо отдаёт простой
        # текст, и TSV не разбирается.
        tail = [
            "-l", self.lang,
            "--oem", str(self.oem),
            "--psm", str(self.psm),
            "--dpi", str(self.dpi),
            "-c", "preserve_interword_spaces=1",
            "-c", "tessedit_create_tsv=1",
        ]
        raw = self._run(image, tail, timeout=180)
        return _parse_tsv(raw)


def _parse_tsv(raw: str) -> OCRResult:
    """Собираем текст из слов TSV и считаем среднюю уверенность (level==5)."""
    lines = raw.splitlines()
    if not lines:
        return OCRResult("", 0.0, 0)

    # Находим строку-заголовок TSV, чтобы отбросить предупреждения tesseract в stderr.
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("level\t") or ln.split("\t")[:1] == ["level"]:
            header_idx = i
            break
    if header_idx is None:
        return OCRResult("", 0.0, 0)

    confs: list[float] = []
    words_by_line: dict[tuple, list[str]] = {}
    order: list[tuple] = []

    for ln in lines[header_idx + 1:]:
        parts = ln.split("\t")
        if len(parts) < 12:
            continue
        try:
            level = int(parts[0])
        except ValueError:
            continue
        if level != 5:
            continue
        text = parts[11]
        conf_raw = parts[10]
        try:
            conf = float(conf_raw)
        except ValueError:
            conf = -1.0
        if text.strip():
            key = (parts[1], parts[2], parts[3], parts[4])  # page/block/par/line
            if key not in words_by_line:
                words_by_line[key] = []
                order.append(key)
            words_by_line[key].append(text)
            if conf >= 0:
                confs.append(conf)

    # Восстанавливаем текст: слова в строку через пробел, строки — через \n.
    text_lines = [" ".join(words_by_line[k]) for k in order]
    text = "\n".join(text_lines)
    mean_conf = round(sum(confs) / len(confs), 2) if confs else 0.0
    return OCRResult(text=text, confidence=mean_conf, n_words=len(confs), lines=text_lines)
