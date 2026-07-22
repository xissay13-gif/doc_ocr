#!/usr/bin/env python3
"""Интерактивное консольное приложение для распознавания судебных приказов.

Как работает:
  * PDF/изображения кладутся в папку  input/  (рядом с программой);
  * при запуске программа спрашивает число потоков;
  * обрабатывает файлы, показывая по каждой странице время и точность;
  * результат (CSV) складывается в папку  output/.

Папку input можно пополнять и повторно запускать проверку, не закрывая программу.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# импорт пакета court_ocr рядом со скриптом/exe
if getattr(sys, "frozen", False):
    BASE = Path(sys.executable).resolve().parent
else:
    BASE = Path(__file__).resolve().parent
    sys.path.insert(0, str(BASE))

from court_ocr.ocr import Tesseract, find_tessdata, find_tesseract   # noqa: E402
from court_ocr.pipeline import Config, run_per_document              # noqa: E402
from court_ocr.render import iter_input_files                        # noqa: E402

INPUT_DIR = BASE / "input"
OUTPUT_DIR = BASE / "output"


def _reconfigure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def ask_threads(max_threads: int) -> int:
    """Спросить число потоков. По умолчанию — автоопределённый максимум (ядра CPU)."""
    print(f"Доступно потоков (ядер) — определено автоматически: {max_threads}.")
    while True:
        raw = input(f"Сколько потоков использовать? [Enter = максимум {max_threads}]: ").strip()
        if not raw:
            return max_threads
        try:
            n = int(raw)
            if 1 <= n <= max(128, max_threads):
                return n
        except ValueError:
            pass
        print(f"  Введите целое число от 1 до {max_threads}.")


def _fmt(sec: float) -> str:
    return f"{sec:.1f}с" if sec < 60 else f"{int(sec // 60)}м {int(sec % 60)}с"


def _mean_conf(rows) -> float:
    vals = [float(r["ocr_confidence"]) for r in rows
            if str(r.get("ocr_confidence", "")).replace(".", "", 1).isdigit()]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def process_batch(files, tess: Tesseract, cfg: Config, threads: int) -> None:
    """Обработать файлы по документам: каждый документ сразу пишет свой PDF + CSV."""
    import time
    print(f"\nФайлов к обработке: {len(files)}. Потоков: {threads}.")
    print("=" * 64)
    t0 = time.perf_counter()

    def on_page(row):
        st = row.get("status", "")
        mark = "OK" if st == "ok" else ("--" if st == "empty" else "!!")
        name = str(row.get("file", ""))[:30]
        line = (f"    {mark} {name} стр.{row.get('page')}: "
                f"{row.get('seconds')}с, точность {row.get('ocr_confidence')}%")
        if st.startswith("error"):
            line += f"  ({st})"
        print(line, flush=True)

    def on_doc(info):
        rows = info["rows"]
        sec = 0.0
        for r in rows:
            try:
                sec += float(r.get("seconds") or 0)
            except ValueError:
                pass
        parts = [info["csv"].name]
        if info["pdf"]:
            parts.append(info["pdf"].name)
        print(f"  ✓ Документ «{info['stem']}» готов: {len(rows)} стр., "
              f"машинное время {_fmt(sec)}, точность {_mean_conf(rows)}%")
        print("      → в папке output: " + "  +  ".join(parts))
        print("-" * 64)

    docs = run_per_document(files, tess, cfg, threads, OUTPUT_DIR,
                            on_page=on_page, on_doc=on_doc)
    elapsed = time.perf_counter() - t0

    all_rows = [r for d in docs for r in d["rows"]]
    errs = sum(1 for r in all_rows if str(r.get("status", "")).startswith("error"))
    print("=" * 64)
    print(f"Готово за {_fmt(elapsed)}. Документов: {len(docs)}, страниц: {len(all_rows)} "
          f"(ошибок: {errs}), средняя точность {_mean_conf(all_rows)}%.")
    print(f"Результаты (CSV + PDF на каждый файл) в папке: {OUTPUT_DIR}")


def main() -> int:
    _reconfigure_console()
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  Распознавание судебных приказов → CSV")
    print("=" * 64)
    print(f"  Папка с PDF (вход):  {INPUT_DIR}")
    print(f"  Папка результатов:   {OUTPUT_DIR}")

    # Движок Tesseract вшит в exe; языковая модель — во внешней папке tessdata/.
    tess_cmd = find_tesseract()
    if not tess_cmd:
        print("\nОШИБКА: не найден движок Tesseract.")
        input("\nНажмите Enter для выхода...")
        return 2

    model_dir = BASE / "tessdata"
    tessdata = str(model_dir) if model_dir.is_dir() else find_tessdata(tess_cmd)
    if not tessdata or not (Path(tessdata) / "rus.traineddata").is_file():
        print(f"\nОШИБКА: не найдена языковая модель.")
        print(f"Положите rus.traineddata и osd.traineddata в папку:\n  {model_dir}")
        input("\nНажмите Enter для выхода...")
        return 2
    print(f"  Языковая модель:     {tessdata}")

    tess = Tesseract(tess_cmd, tessdata=tessdata, lang="rus", psm=6, oem=1, dpi=300)
    cfg = Config(dpi=300, keep_raw=True, make_pdf=True, base_dir=INPUT_DIR)

    print()
    threads = ask_threads(os.cpu_count() or 4)

    processed: set = set()  # (путь, размер, время изменения) — чтобы не гонять дважды
    while True:
        all_files = list(iter_input_files([INPUT_DIR], recursive=True))
        new_files = []
        for f in all_files:
            try:
                key = (str(f), f.stat().st_size, int(f.stat().st_mtime))
            except OSError:
                continue
            if key not in processed:
                new_files.append(f)
                processed.add(key)

        if new_files:
            process_batch(new_files, tess, cfg, threads)
        elif all_files:
            print(f"\nНовых файлов нет (уже обработано: {len(all_files)}).")
        else:
            print(f"\nВ папке {INPUT_DIR} нет PDF/изображений.")

        print()
        ans = input("Проверить папку снова? [Enter — да, q — выход]: ").strip().lower()
        if ans in ("q", "quit", "exit", "выход", "в"):
            break

    print("Завершение.")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:  # чтобы окно не закрылось молча при ошибке
        import traceback
        traceback.print_exc()
        try:
            input("\nПроизошла ошибка. Нажмите Enter для выхода...")
        except EOFError:
            pass
        code = 1
    raise SystemExit(code)
