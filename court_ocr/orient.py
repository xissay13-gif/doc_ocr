"""Определение и исправление поворота страницы.

Стратегия (без OpenCV, только numpy/scipy/PIL):

1. OSD Tesseract (``--psm 0``) уверенно различает 0/90/180/270, включая
   «вверх ногами» (180°). Кратные 90° применяем без интерполяции (transpose).
2. После этого — узкий поиск перекоса проекционным профилем (±10°) убирает
   остаточный наклон сканера.
3. Если OSD не уверен или упал (типичная «подпись» листа, повёрнутого на ~45°) —
   широкий поиск профилем в диапазоне ±46°: угол, при котором строки текста
   становятся горизонтальными, даёт максимум дисперсии суммы по строкам. Затем
   повторный OSD на выровненном листе снимает неоднозначность «прямо/вверх ногами».

Важно: и оценка угла, и его применение используют один и тот же
``scipy.ndimage.rotate`` над 2D-массивом (grayscale), поэтому знак угла совпадает
и наклон не «удваивается».
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import rotate as nd_rotate

from .ocr import Tesseract

# Порог уверенности OSD: ниже — считаем ориентацию не определённой.
OSD_CONF_MIN = 1.0


@dataclass
class OrientResult:
    image: Image.Image
    rotation: str      # человекочитаемое описание применённого поворота
    method: str        # osd | wide | deskew-only | none


# --------------------------------------------------------------------------- #
# Бинаризация (Otsu на чистом numpy) — для оценки угла по профилю
# --------------------------------------------------------------------------- #
def _otsu_binary(gray: np.ndarray) -> np.ndarray:
    """Тёмные пиксели (текст) → 1.0, фон → 0.0."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return np.zeros_like(gray, dtype=np.float32)
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-9
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    t = int(np.nanargmax(sigma_b))
    return (gray < t).astype(np.float32)


def _skew_score(binary: np.ndarray, angle: float) -> float:
    """Дисперсия суммы по строкам после поворота на angle (больше — ровнее)."""
    rot = nd_rotate(binary, angle, reshape=False, order=0, mode="constant", cval=0.0)
    rows = rot.sum(axis=1)
    return float(rows.var())


def find_skew(
    binary: np.ndarray,
    lo: float = -10.0,
    hi: float = 10.0,
    step: float = 1.0,
    fine: float = 1.0,
    fstep: float = 0.1,
) -> float:
    """Ищем корректирующий угол: сначала грубо, потом уточняем вокруг максимума."""
    small = binary[::4, ::4]
    if small.size == 0 or small.sum() == 0:
        return 0.0
    coarse = np.arange(lo, hi + step / 2, step)
    best = max(coarse, key=lambda a: _skew_score(small, a))
    fine_grid = np.arange(best - fine, best + fine + fstep / 2, fstep)
    best = max(fine_grid, key=lambda a: _skew_score(small, a))
    return float(best)


# --------------------------------------------------------------------------- #
# Применение поворота
# --------------------------------------------------------------------------- #
# rotate (по часовой, из OSD) → transpose PIL (против часовой), без интерполяции.
_CARDINAL = {90: Image.ROTATE_270, 180: Image.ROTATE_180, 270: Image.ROTATE_90}


def apply_cardinal(image: Image.Image, rotate_cw: int) -> Image.Image:
    op = _CARDINAL.get(rotate_cw % 360)
    return image.transpose(op) if op is not None else image


def apply_fine(gray: Image.Image, angle: float) -> Image.Image:
    """Дробный поворот с белым фоном и без обрезки углов (bicubic)."""
    if abs(angle) < 0.05:
        return gray
    arr = np.asarray(gray)
    out = nd_rotate(arr, angle, reshape=True, order=3, mode="constant", cval=255)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# --------------------------------------------------------------------------- #
# Основная функция
# --------------------------------------------------------------------------- #
def _wide_correct(gray: Image.Image, tess: Optional[Tesseract], wide: float):
    """Широкий поиск угла (±wide) + повторный OSD для снятия флипа. → (img, desc)."""
    binary = _otsu_binary(np.asarray(gray))
    angle = find_skew(binary, -wide, wide, 1.0)
    img = apply_fine(gray, angle)
    desc = f"широкий {angle:+.1f}°"
    if tess is not None:
        osd2 = tess.osd(img)
        if (osd2 is not None and osd2.confidence is not None
                and osd2.confidence >= OSD_CONF_MIN and osd2.rotate in (90, 180, 270)):
            img = apply_cardinal(img, osd2.rotate)
            desc += f" + OSD {osd2.rotate}°"
    return img, desc


def correct_orientation(
    gray: Image.Image,
    tess: Optional[Tesseract],
    max_skew: float = 10.0,
    wide: float = 46.0,
) -> OrientResult:
    """Возвращаем выровненное изображение и описание применённого поворота."""
    # Без Tesseract остаётся только выравнивание наклона (флип не определить).
    if tess is None:
        img, desc = _wide_correct(gray, None, wide)
        return OrientResult(img, desc.replace("широкий", "перекос"), "deskew-only")

    osd = tess.osd(gray)
    good = (
        osd is not None
        and osd.confidence is not None
        and osd.confidence >= OSD_CONF_MIN
        and osd.rotate in (0, 90, 180, 270)
    )

    if good:
        img = apply_cardinal(gray, osd.rotate) if osd.rotate else gray
        binary = _otsu_binary(np.asarray(img))
        angle = find_skew(binary, -max_skew, max_skew, 1.0)
        # Наклон упёрся в границу узкого окна → OSD «промахнулся» (типично для ~45°,
        # который OSD ошибочно принимает за 0° со средней уверенностью). Широкий поиск.
        if abs(angle) >= max_skew - 0.5:
            img2, desc2 = _wide_correct(img, tess, wide)
            base = f"OSD {osd.rotate}°" if osd.rotate else "OSD 0°"
            return OrientResult(img2, f"{base} → {desc2}", "osd+wide")
        img = apply_fine(img, angle)
        desc = (f"OSD {osd.rotate}°" if osd.rotate else "OSD 0°")
        if abs(angle) >= 0.05:
            desc += f" + перекос {angle:+.1f}°"
        return OrientResult(img, desc, "osd")

    # OSD не уверен → вероятен поворот ~45°: широкий поиск профилем.
    img, desc = _wide_correct(gray, tess, wide)
    return OrientResult(img, desc, "wide")
