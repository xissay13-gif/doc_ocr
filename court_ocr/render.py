"""Рендер страниц PDF в изображения через PyMuPDF (poppler не требуется).

Также умеет обрабатывать входные изображения (PNG/JPG/TIFF) как одну «страницу».
Возвращаются grayscale-изображения PIL — в таком виде их удобно и выравнивать,
и распознавать.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter, ImageOps

# MuPDF исторически не потокобезопасен. Рендер PDF быстрый (основное время — OCR),
# поэтому сериализуем именно рендер: качество распараллеливания это почти не меняет,
# зато исключает нативные падения при большом числе потоков.
_FITZ_LOCK = threading.Lock()

PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTS = PDF_EXTS | IMAGE_EXTS


def count_pages(path: Path) -> int:
    """Число «страниц» в файле (для изображений — 1)."""
    path = Path(path)
    if path.suffix.lower() in PDF_EXTS:
        with fitz.open(path) as doc:
            return doc.page_count
    return 1


def render_page(path: Path, page_index: int, dpi: int = 300) -> Image.Image:
    """Рендер одной страницы в grayscale PIL. Документ открывается на вызов
    (объекты PyMuPDF не потокобезопасны при совместном использовании)."""
    path = Path(path)
    if path.suffix.lower() in PDF_EXTS:
        with _FITZ_LOCK:
            with fitz.open(path) as doc:
                page = doc.load_page(page_index)
                zoom = dpi / 72.0
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                                      colorspace=fitz.csGRAY, alpha=False)
                # копируем в отдельный массив, пока держим блокировку (samples —
                # представление буфера pixmap; отвязываемся от объекта MuPDF)
                arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
        return Image.fromarray(arr, mode="L")
    # Обычная картинка.
    with Image.open(path) as im:
        return im.convert("L")


def preprocess(gray: Image.Image, binarize: bool = False, denoise: bool = True,
               pad: int = 15) -> Image.Image:
    """Лёгкая подготовка под OCR: контраст, (опц.) бинаризация, шумоподавление, поля."""
    out = ImageOps.autocontrast(gray)
    if denoise:
        out = out.filter(ImageFilter.MedianFilter(3))
    if binarize:
        arr = np.asarray(out)
        hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        total = hist.sum()
        p = hist / total
        omega = np.cumsum(p)
        mu = np.cumsum(p * np.arange(256))
        mu_t = mu[-1]
        denom = omega * (1.0 - omega)
        denom[denom == 0] = 1e-9
        sigma_b = (mu_t * omega - mu) ** 2 / denom
        t = int(np.nanargmax(sigma_b))
        arr = np.where(arr > t, 255, 0).astype(np.uint8)
        out = Image.fromarray(arr)
    if pad:
        out = ImageOps.expand(out, border=pad, fill=255)
    return out


def iter_tasks(paths, recursive: bool = False) -> Iterator[Tuple[Path, int, int]]:
    """Перебор всех задач (файл, индекс_страницы, всего_страниц) по входным путям."""
    for path in iter_input_files(paths, recursive):
        try:
            n = count_pages(path)
        except Exception:
            n = 0
        for i in range(n):
            yield path, i, n


def iter_input_files(paths, recursive: bool = False) -> Iterator[Path]:
    """Разворачиваем список путей (файлы/папки) в поддерживаемые файлы."""
    for p in paths:
        path = Path(p)
        if path.is_dir():
            globber = path.rglob("*") if recursive else path.glob("*")
            for f in sorted(globber):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                    yield f
        elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path
