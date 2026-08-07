#!/usr/bin/env python3
"""OCR судебных приказов (и подобных сканов) → CSV.

Возможности:
  * рендер PDF без poppler (PyMuPDF) и обычные изображения (PNG/JPG/TIFF);
  * встроенный Tesseract (папка ./tesseract рядом со скриптом) — установка не нужна;
  * авто-исправление поворота на 90/180/270° и на ~45° (перекос сканера тоже);
  * распознавание rus+eng, извлечение полей приказа регулярными выражениями;
  * выбираемое число потоков (-j/--threads);
  * результат — один CSV (page;line_id;line, кодировка Windows-1251, «;»);
  * режим --watch: непрерывно следить за папкой и распознавать документы сразу,
    как только они в неё попали (результат — CSV + PDF на каждый документ).

Примеры:
  python ocr_court.py "C:\\сканы"                 обработать папку
  python ocr_court.py doc.pdf -o out.csv -j 8    один файл, 8 потоков
  python ocr_court.py "C:\\сканы" -r --save-text txt   рекурсивно + сохранить текст
  python ocr_court.py "C:\\сканы" --watch --output-dir "D:\\OCR"   мониторинг папки

Если Tesseract не встроен — сначала выполните:  python setup_tesseract.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# гарантируем импорт пакета court_ocr рядом со скриптом
sys.path.insert(0, str(Path(__file__).resolve().parent))

from court_ocr import __version__                     # noqa: E402
from court_ocr.ocr import Tesseract, find_tessdata, find_tesseract  # noqa: E402
from court_ocr.pipeline import (Config, page_rows_to_lines, run,     # noqa: E402
                                run_per_document, write_lines_csv)
from court_ocr.render import iter_tasks                # noqa: E402
from court_ocr.watch import (DEFAULT_INTERVAL, DEFAULT_STABLE,  # noqa: E402
                             STATE_NAME, StateStore, Watcher)


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

    w = p.add_argument_group("непрерывный мониторинг папки")
    w.add_argument("--watch", action="store_true",
                   help="не завершаться: следить за папками и распознавать новые файлы сразу")
    w.add_argument("--output-dir", metavar="DIR", default="output",
                   help="папка результатов в режиме --watch: на каждый документ "
                        "свой CSV и PDF (по умолчанию ./output)")
    w.add_argument("--watch-interval", type=float, default=DEFAULT_INTERVAL,
                   metavar="SEC", help=f"период опроса папки, с (по умолчанию {DEFAULT_INTERVAL:g})")
    w.add_argument("--watch-stable", type=int, default=DEFAULT_STABLE, metavar="N",
                   help="сколько опросов подряд файл должен быть неизменным, чтобы "
                        f"считаться дописанным (по умолчанию {DEFAULT_STABLE})")
    w.add_argument("--watch-new-only", action="store_true",
                   help="не трогать файлы, которые уже лежат в папке на момент запуска")
    w.add_argument("--no-pdf", action="store_true",
                   help="в режиме --watch не создавать распознанный PDF (только CSV)")
    w.add_argument("--rescan", action="store_true",
                   help="забыть обработанное (сбросить память в папке результатов)")
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

    # Сформировать список задач (файл, страница). В режиме --watch пустая папка
    # на старте — нормальная ситуация: файлы появятся позже.
    tasks = []
    if not args.watch:
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
        make_pdf=args.watch and not args.no_pdf,
        keep_raw=not args.no_raw_text,
        delimiter=args.delimiter,
        base_dir=base_dir,
        save_text_dir=Path(args.save_text) if args.save_text else None,
    )

    sys.stderr.write(
        f"Tesseract: {tess_cmd}\n"
        f"tessdata:  {tessdata or '(по умолчанию)'}\n"
    )

    if args.watch:
        return _watch(args, tess, cfg)

    sys.stderr.write(f"Потоков:   {args.threads}   Страниц: {len(tasks)}\n")

    rows = run(tasks, tess, cfg, threads=args.threads, progress=True)

    # Построчный CSV: page; line_id; line (кодировка Windows-1251).
    line_rows = page_rows_to_lines(rows)
    out_path = Path(args.output)
    write_lines_csv(line_rows, out_path, delimiter=args.delimiter, encoding="cp1251")

    _summary(rows, out_path, len(line_rows))
    return 0


def _watch(args, tess: Tesseract, cfg: Config) -> int:
    """Непрерывный мониторинг: распознаём документы по мере появления в папке.

    Результат пишется по документам (как в интерактивном приложении): на каждый
    входной файл — свой CSV и, если не отключено, распознанный PDF."""
    output_dir = Path(args.output_dir).expanduser()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(f"Не удалось создать папку результатов {output_dir}: {exc}\n")
        return 2

    state = StateStore(output_dir / STATE_NAME)
    if args.rescan:
        state.clear()

    watcher = Watcher(
        [Path(p) for p in args.inputs],
        recursive=args.recursive,
        interval=args.watch_interval,
        stable_checks=args.watch_stable,
        exclude=[output_dir],   # распознанные PDF не должны вернуться на вход
        state=state,
    )
    if args.watch_new_only:
        sys.stderr.write(f"Пропущено файлов, уже лежащих в папке: {watcher.skip_existing()}\n")

    sys.stderr.write(
        f"Потоков:   {args.threads}\n"
        f"Слежу за:  {', '.join(str(Path(p)) for p in args.inputs)}"
        f"{' (с подпапками)' if args.recursive else ''}\n"
        f"Результат: {output_dir.resolve()} — "
        f"CSV{' + PDF' if cfg.make_pdf else ''} на каждый документ\n"
        f"Опрос раз в {args.watch_interval:g} с. Ctrl+C — остановить.\n"
    )

    totals = {"docs": 0, "pages": 0, "lines": 0, "errors": 0}

    def on_page(row: dict) -> None:
        status = str(row.get("status", ""))
        totals["pages"] += 1
        if status.startswith("error"):
            totals["errors"] += 1
            sys.stderr.write(f"  [!] {row.get('file')} стр.{row.get('page')}: {status}\n")

    def on_doc(info: dict) -> None:
        totals["docs"] += 1
        totals["lines"] += info["n_lines"]
        names = info["csv"].name + (f" + {info['pdf'].name}" if info["pdf"] else "")
        print(f"[{time.strftime('%H:%M:%S')}] {info['stem']}: "
              f"{len(info['pages'])} стр., {info['n_lines']} строк → {names}", flush=True)

    def handler(files) -> None:
        sys.stderr.write(f"\nНовых файлов: {len(files)} — распознаю...\n")
        run_per_document(files, tess, cfg, args.threads, output_dir,
                         on_page=on_page, on_doc=on_doc)

    try:
        watcher.run(handler)
    except KeyboardInterrupt:
        sys.stderr.write(
            "\nМониторинг остановлен.\n"
            f"  Документов: {totals['docs']}, страниц: {totals['pages']} "
            f"(ошибок: {totals['errors']}), строк: {totals['lines']}\n"
            f"  Результаты: {output_dir.resolve()}\n"
        )
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
