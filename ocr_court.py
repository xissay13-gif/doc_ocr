#!/usr/bin/env python3
"""OCR судебных приказов (и подобных сканов) → CSV.

Возможности:
  * рендер PDF без poppler (PyMuPDF) и обычные изображения (PNG/JPG/TIFF);
  * встроенный Tesseract (папка ./tesseract рядом со скриптом) — установка не нужна;
  * авто-исправление поворота на 90/180/270° и на ~45° (перекос сканера тоже);
  * распознавание rus+eng, извлечение полей приказа регулярными выражениями;
  * выбираемое число потоков (-j/--threads);
  * результат — один CSV (UTF-8 BOM, разделитель «;» — открывается в Excel).

Примеры:
  python ocr_court.py "C:\\сканы"                 обработать папку
  python ocr_court.py doc.pdf -o out.csv -j 8    один файл, 8 потоков
  python ocr_court.py "C:\\сканы" -r --save-text txt   рекурсивно + сохранить текст

Если Tesseract не встроен — сначала выполните:  python setup_tesseract.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# гарантируем импорт пакета court_ocr рядом со скриптом
sys.path.insert(0, str(Path(__file__).resolve().parent))

from court_ocr import __version__                     # noqa: E402
from court_ocr.ocr import Tesseract, find_tessdata, find_tesseract  # noqa: E402
from court_ocr.pipeline import (Config, page_rows_to_lines, run,     # noqa: E402
                                write_lines_csv)
from court_ocr.render import iter_tasks                # noqa: E402


def _reconfigure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ocr_court.py",
        description="OCR сканов судебных приказов в CSV (встроенный Tesseract, потоки, авто-поворот).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help="PDF/изображения или папки с ними")
    p.add_argument("-o", "--output", default="court_ocr_result.csv",
                   help="файл CSV (по умолчанию court_ocr_result.csv)")
    p.add_argument("-j", "--threads", type=int, default=os.cpu_count() or 4,
                   help="число потоков (по умолчанию = число ядер)")
    p.add_argument("-r", "--recursive", action="store_true",
                   help="искать файлы в подпапках")
    p.add_argument("--dpi", type=int, default=300, help="DPI рендера PDF (по умолчанию 300)")
    p.add_argument("--binarize", action="store_true",
                   help="бинаризация (Otsu) перед OCR — иногда точнее на грязных сканах")
    p.add_argument("--max-skew", type=float, default=10.0,
                   help="макс. остаточный перекос после OSD, ° (по умолчанию 10)")
    p.add_argument("--wide", type=float, default=46.0,
                   help="диапазон широкого поиска угла для ~45°, ° (по умолчанию 46)")
    p.add_argument("--delimiter", default=";", help="разделитель CSV (по умолчанию «;»)")
    p.add_argument("--no-raw-text", action="store_true",
                   help="не сохранять колонку с полным распознанным текстом")
    p.add_argument("--save-text", metavar="DIR",
                   help="сохранять распознанный текст каждой страницы в .txt в этой папке")
    p.add_argument("--tesseract", help="путь к tesseract.exe (если не встроен и не в PATH)")
    p.add_argument("--tessdata-dir", metavar="DIR",
                   help="папка с моделями .traineddata (по умолчанию встроенная tesseract/tessdata)")
    p.add_argument("--lang", default="rus", help="языки Tesseract (по умолчанию rus)")
    p.add_argument("--psm", type=int, default=6, help="Tesseract --psm (по умолчанию 6)")
    p.add_argument("--oem", type=int, default=1, help="Tesseract --oem (по умолчанию 1, LSTM)")
    p.add_argument("--list", action="store_true", help="только перечислить найденные файлы")
    p.add_argument("--version", action="version", version=f"court-ocr {__version__}")
    return p


def main(argv=None) -> int:
    _reconfigure_console()
    args = build_parser().parse_args(argv)

    # csv требует разделитель ровно из одного символа (иначе падение уже ПОСЛЕ
    # всего OCR). Разрешаем удобную запись «\t» для табуляции.
    if args.delimiter == "\\t":
        args.delimiter = "\t"
    if len(args.delimiter) != 1:
        sys.stderr.write("Разделитель (--delimiter) должен быть одним символом.\n")
        return 2

    # Сформировать список задач (файл, страница).
    tasks = list(iter_tasks(args.inputs, recursive=args.recursive))
    if not tasks:
        sys.stderr.write("Не найдено ни одного PDF/изображения по указанным путям.\n")
        return 1

    if args.list:
        files = sorted({t[0] for t in tasks})
        for f in files:
            print(f)
        print(f"\nВсего файлов: {len(files)}, страниц: {len(tasks)}")
        return 0

    # Найти Tesseract.
    tess_cmd = find_tesseract(args.tesseract)
    if not tess_cmd:
        sys.stderr.write(
            "Tesseract не найден.\n"
            "  • встройте его командой:  python setup_tesseract.py\n"
            "  • либо укажите путь:      --tesseract \"C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe\"\n"
            "  • либо установите в PATH.\n"
        )
        return 2

    if args.tessdata_dir:
        tessdata = args.tessdata_dir
        if not Path(tessdata).is_dir():
            sys.stderr.write(f"Папка с моделями не найдена: {tessdata}\n")
            return 2
    else:
        tessdata = find_tessdata(tess_cmd)
    tess = Tesseract(tess_cmd, tessdata=tessdata, lang=args.lang, psm=args.psm,
                     oem=args.oem, dpi=args.dpi)

    # База для относительных имён: если единственный вход — папка.
    base_dir = None
    if len(args.inputs) == 1 and Path(args.inputs[0]).is_dir():
        base_dir = Path(args.inputs[0])

    cfg = Config(
        dpi=args.dpi,
        binarize=args.binarize,
        max_skew=args.max_skew,
        wide=args.wide,
        keep_raw=not args.no_raw_text,
        base_dir=base_dir,
        save_text_dir=Path(args.save_text) if args.save_text else None,
    )

    sys.stderr.write(
        f"Tesseract: {tess_cmd}\n"
        f"tessdata:  {tessdata or '(по умолчанию)'}\n"
        f"Потоков:   {args.threads}   Страниц: {len(tasks)}\n"
    )

    rows = run(tasks, tess, cfg, threads=args.threads, progress=True)

    # Построчный CSV: page; line_id; line (кодировка Windows-1251).
    line_rows = page_rows_to_lines(rows)
    out_path = Path(args.output)
    write_lines_csv(line_rows, out_path, delimiter=args.delimiter, encoding="cp1251")

    _summary(rows, out_path, len(line_rows))
    return 0


def _summary(rows, out_path: Path, n_lines: int) -> None:
    ok = sum(1 for r in rows if r.get("status") == "ok")
    empty = sum(1 for r in rows if r.get("status") == "empty")
    errors = sum(1 for r in rows if str(r.get("status", "")).startswith("error"))
    confs = [float(r["ocr_confidence"]) for r in rows
             if str(r.get("ocr_confidence", "")).replace(".", "", 1).isdigit()]
    mean_conf = round(sum(confs) / len(confs), 1) if confs else 0.0
    rotated = sum(1 for r in rows if r.get("rotation_deg") and
                  r["rotation_deg"] not in ("OSD 0°", ""))
    sys.stderr.write(
        "\nГотово.\n"
        f"  CSV (page;line_id;line, cp1251): {out_path.resolve()}\n"
        f"  Страниц:        {len(rows)} (ok={ok}, пусто={empty}, ошибок={errors}), строк: {n_lines}\n"
        f"  Средняя увер.:  {mean_conf}%\n"
        f"  Развёрнуто:     {rotated} страниц\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
