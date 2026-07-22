"""Конвейер обработки страниц в несколько потоков и запись результата в CSV."""
from __future__ import annotations

import csv
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from PIL import ImageOps

from . import extract
from .ocr import Tesseract
from .orient import correct_orientation
from .render import preprocess, render_page


@dataclass
class Config:
    dpi: int = 300
    binarize: bool = False
    max_skew: float = 10.0
    wide: float = 46.0
    keep_raw: bool = True
    make_pdf: bool = False                 # создавать ли searchable-PDF
    base_dir: Optional[Path] = None       # для относительных имён в колонке file
    save_text_dir: Optional[Path] = None  # куда класть .txt каждой страницы


_print_lock = threading.Lock()

# Поля, которыми управляет конвейер (не должны затираться пустыми значениями из
# extract_fields, который возвращает весь набор FIELDS с "" по умолчанию).
_PIPELINE_KEYS = {
    "file", "page", "rotation_deg", "ocr_confidence", "ocr_chars", "seconds",
    "status", "raw_text",
}


def _rel(path: Path, base: Optional[Path]) -> str:
    if base:
        try:
            return str(path.relative_to(base))
        except ValueError:
            pass
    return path.name


def process_page(task, tess: Optional[Tesseract], cfg: Config):
    """Обработать одну страницу. Вернуть (row, pdf_bytes).

    pdf_bytes — одностраничный searchable-PDF (если cfg.make_pdf), иначе b""."""
    path, page_index, _n = task
    row = {k: "" for k in extract.FIELDS}
    row["file"] = _rel(path, cfg.base_dir)
    row["page"] = page_index + 1
    pdf_bytes = b""
    start = time.perf_counter()
    try:
        gray = render_page(path, page_index, dpi=cfg.dpi)
        base = ImageOps.autocontrast(gray)

        orient = correct_orientation(base, tess, max_skew=cfg.max_skew, wide=cfg.wide)
        row["rotation_deg"] = orient.rotation

        if tess is None:
            row["status"] = "no_tesseract"
            row["seconds"] = round(time.perf_counter() - start, 2)
            return row, pdf_bytes

        ocr_img = preprocess(orient.image, binarize=cfg.binarize, denoise=True, pad=15)
        if cfg.make_pdf:
            result, pdf_bytes = tess.image_to_document(ocr_img)
        else:
            result = tess.image_to_data(ocr_img)

        fields = extract.extract_fields(result.text)
        for k, v in fields.items():
            if k in _PIPELINE_KEYS:
                continue
            row[k] = v

        row["ocr_confidence"] = result.confidence
        row["ocr_chars"] = len(result.text)
        row["status"] = "ok" if result.text.strip() else "empty"
        row["raw_text"] = result.text if cfg.keep_raw else ""

        if cfg.save_text_dir:
            _save_text(cfg.save_text_dir, path, page_index, result.text)
    except Exception as exc:  # одна страница не должна ронять весь прогон
        row["status"] = f"error: {type(exc).__name__}: {exc}"
        row["rotation_deg"] = row.get("rotation_deg", "")
        with _print_lock:  # не перебиваем строку прогресса главного потока
            sys.stderr.write(
                f"\n[!] Ошибка на {row['file']} стр.{page_index + 1}: {exc}\n"
                + traceback.format_exc()
            )
    row["seconds"] = round(time.perf_counter() - start, 2)
    return row, pdf_bytes


def _save_text(out_dir: Path, path: Path, page_index: int, text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{path.stem}_p{page_index + 1:03d}.txt"
    (out_dir / name).write_text(text, encoding="utf-8")


def run(tasks: List, tess: Optional[Tesseract], cfg: Config, threads: int,
        progress: bool = True,
        on_page: Optional[Callable[[dict, int, int], None]] = None) -> List[dict]:
    """Обработать все задачи в пуле из `threads` потоков, вернуть строки CSV.

    on_page(row, done, total) вызывается по мере готовности каждой страницы
    (под общей блокировкой печати) — для показа времени и точности.
    """
    rows: List[dict] = []
    total = len(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = {pool.submit(process_page, t, tess, cfg): t for t in tasks}
        for fut in as_completed(futures):
            row, _pdf = fut.result()
            rows.append(row)
            done += 1
            if on_page is not None:
                with _print_lock:
                    on_page(row, done, total)
            elif progress:
                with _print_lock:
                    sys.stderr.write(f"\rОбработано {done}/{total} страниц...")
                    sys.stderr.flush()
    if progress and total and on_page is None:
        sys.stderr.write("\n")
    # Упорядочим по файлу и странице для стабильного CSV.
    rows.sort(key=lambda r: (str(r.get("file", "")), int(r.get("page", 0) or 0)))
    return rows


def _safe_stem(path: Path) -> str:
    """Имя файла без расширения, очищенное от недопустимых для Windows символов."""
    stem = path.stem
    for ch in '<>:"/\\|?*':
        stem = stem.replace(ch, "_")
    return stem.strip() or "документ"


def _merge_pdfs(page_pdfs: List[bytes], out_path: Path) -> bool:
    """Склеить одностраничные PDF (в порядке страниц) в один. Возвращает True при успехе."""
    import fitz  # PyMuPDF

    from .render import _FITZ_LOCK  # fitz не потокобезопасен — берём общую блокировку
    with _FITZ_LOCK:
        out = fitz.open()
        try:
            for data in page_pdfs:
                if not data:
                    continue
                src = fitz.open(stream=data, filetype="pdf")
                out.insert_pdf(src)
                src.close()
            if out.page_count:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out.save(str(out_path))
                return True
            return False
        finally:
            out.close()


def write_document(src_path: Path, page_map: dict, output_dir: Path,
                   keep_raw: bool = True, delimiter: str = ";",
                   make_pdf: bool = True) -> dict:
    """Записать результат ОДНОГО документа: <имя>.csv (+ <имя>.pdf) прямо в output/."""
    stem = _safe_stem(src_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(page_map.keys())
    rows = [page_map[i][0] for i in pages]

    csv_path = output_dir / f"{stem}.csv"
    write_csv(rows, csv_path, delimiter=delimiter, keep_raw=keep_raw)

    pdf_path = None
    if make_pdf:
        page_pdfs = [page_map[i][1] for i in pages]
        target = output_dir / f"{stem}.pdf"
        if _merge_pdfs(page_pdfs, target):
            pdf_path = target

    return {"stem": stem, "dir": output_dir, "csv": csv_path, "pdf": pdf_path, "rows": rows}


def run_per_document(files: List[Path], tess: Optional[Tesseract], cfg: Config,
                     threads: int, output_dir: Path,
                     on_page: Optional[Callable[[dict], None]] = None,
                     on_doc: Optional[Callable[[dict], None]] = None) -> List[dict]:
    """Обработать файлы по страницам в общем пуле из `threads` потоков.

    Как только все страницы очередного документа готовы — его CSV и searchable-PDF
    сразу пишутся в output/<имя>/ и вызывается on_doc(info). on_page(row) — по
    каждой странице. Возвращает список info по документам (в порядке готовности)."""
    from .render import count_pages

    tasks = []
    remaining: dict = {}
    collected: dict = {}
    for f in files:
        f = Path(f)
        try:
            n = count_pages(f)
        except Exception:
            n = 0
        if n <= 0:
            continue
        remaining[f] = n
        collected[f] = {}
        for i in range(n):
            tasks.append((f, i, n))

    docs_info: List[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = {pool.submit(process_page, t, tess, cfg): t for t in tasks}
        for fut in as_completed(futures):
            (f, page_index, _n) = futures[fut]
            row, pdf_bytes = fut.result()
            collected[f][page_index] = (row, pdf_bytes)
            if on_page is not None:
                with _print_lock:
                    on_page(row)
            remaining[f] -= 1
            if remaining[f] == 0:
                info = write_document(
                    f, collected[f], output_dir,
                    keep_raw=cfg.keep_raw, make_pdf=cfg.make_pdf,
                )
                docs_info.append(info)
                if on_doc is not None:
                    with _print_lock:
                        on_doc(info)
                collected[f] = {}  # освобождаем память (PDF-байты страниц)
    return docs_info


def write_csv(rows: List[dict], out_path: Path, delimiter: str = ";",
              keep_raw: bool = True) -> None:
    """Записать строки в CSV (UTF-8 с BOM — чтобы Excel корректно открыл кириллицу)."""
    fields = list(extract.FIELDS)
    if not keep_raw and "raw_text" in fields:
        fields.remove("raw_text")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, delimiter=delimiter, extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
