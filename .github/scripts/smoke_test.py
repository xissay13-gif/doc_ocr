"""Проверка собранной программы на настоящем скане.

Мало убедиться, что exe запускается: внутрь вшит движок Tesseract, а модель
лежит рядом в tessdata/ — сломаться может любая из этих связей, и заметно это
только на реальном распознавании. Поэтому скрипт рисует лист с русским текстом,
скармливает его собранной программе и проверяет, что в CSV появился текст, а
рядом — распознанный PDF.

Запуск:  python .github/scripts/smoke_test.py package/court-ocr/court-ocr.exe
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LINES = [
    "Судебный участок № 57 в Ленинском судебном районе",
    "СУДЕБНЫЙ ПРИКАЗ",
    "Дело № 2-1234/2024              05 апреля 2024 года",
    "Взыскать задолженность за отопление в размере",
    "14554,76 руб. и госпошлину 315 руб. 10 коп.",
]
# Слово, которое обязано найтись в CSV: короткое, заглавное, без сложных лигатур.
MUST_CONTAIN = "ПРИКАЗ"
FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONTS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    raise SystemExit("не найден шрифт с кириллицей для генерации тестового скана")


def make_scan(path: Path) -> None:
    """Нарисовать лист А4 (200 dpi) с текстом судебного приказа."""
    img = Image.new("L", (1654, 2339), 255)
    draw = ImageDraw.Draw(img)
    font = _font(42)
    y = 220
    for line in LINES:
        draw.text((160, y), line, fill=20, font=font)
        y += 90
    img.save(path)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    program = Path(argv[1]).resolve()
    if not program.is_file():
        print(f"ОШИБКА: не найдена программа {program}")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inbox, outbox = root / "in", root / "out"
        inbox.mkdir()
        make_scan(inbox / "тестовый_приказ.png")

        cmd = [str(program), "--input", str(inbox), "--output", str(outbox),
               "--threads", "2", "--once", "--no-menu"]
        print("Запускаю:", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
        out = proc.stdout.decode("utf-8", "replace")
        err = proc.stderr.decode("utf-8", "replace")
        print(out)
        if err.strip():
            print("stderr:", err)

        if proc.returncode != 0:
            print(f"ОШИБКА: программа завершилась с кодом {proc.returncode}")
            return 1

        csv_path = outbox / "тестовый_приказ.csv"
        pdf_path = outbox / "тестовый_приказ.pdf"
        if not csv_path.is_file():
            print(f"ОШИБКА: не создан CSV. В папке результатов: "
                  f"{[p.name for p in outbox.glob('*')] if outbox.is_dir() else 'папки нет'}")
            return 1

        text = csv_path.read_text(encoding="cp1251", errors="replace")
        rows = [ln for ln in text.splitlines()[1:] if ln.strip()]
        print(f"CSV: {len(rows)} строк\n" + "\n".join(f"  {r}" for r in rows[:10]))

        if len(rows) < 3:
            print(f"ОШИБКА: распознано слишком мало строк ({len(rows)}) — "
                  f"похоже, движок или языковая модель не подхватились")
            return 1
        if MUST_CONTAIN not in text:
            print(f"ОШИБКА: в распознанном тексте нет слова «{MUST_CONTAIN}»")
            return 1

        if not pdf_path.is_file():
            print("ОШИБКА: не создан распознанный PDF")
            return 1
        if pdf_path.read_bytes()[:5] != b"%PDF-":
            print("ОШИБКА: созданный PDF повреждён")
            return 1
        print(f"PDF: {pdf_path.stat().st_size // 1024} КБ")

    print("\nПроверка пройдена: движок, языковая модель, CSV и PDF работают.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
