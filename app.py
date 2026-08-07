#!/usr/bin/env python3
"""Интерактивное консольное приложение для распознавания судебных приказов.

Два режима работы:

  * **непрерывный мониторинг** — программа следит за входной папкой и
    распознаёт каждый документ сразу, как только он туда попал (Ctrl+C — стоп);
  * **разовый проход** — обработать то, что лежит в папке, и вернуться в меню.

Пути к папкам гибко настраиваются (позднее перекрывает раннее):

    умолчания (input/ и output/ рядом с программой)
      → court-ocr.json рядом с программой
      → переменные окружения COURT_OCR_INPUT / COURT_OCR_OUTPUT / ...
      → ключи командной строки  --input / --output

Примеры запуска:

    court-ocr.exe                                   меню
    court-ocr.exe --input D:\\Сканы --watch --no-menu   мониторинг без вопросов
    court-ocr.exe -i \\\\server\\scan -o D:\\OCR --once   разовый проход
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# импорт пакета court_ocr рядом со скриптом/exe
if getattr(sys, "frozen", False):
    BASE = Path(sys.executable).resolve().parent
else:
    BASE = Path(__file__).resolve().parent
    sys.path.insert(0, str(BASE))

from court_ocr import settings                                     # noqa: E402
from court_ocr.ocr import Tesseract, find_tessdata, find_tesseract  # noqa: E402
from court_ocr.pipeline import Config, run_per_document             # noqa: E402
from court_ocr.render import iter_input_files                       # noqa: E402
from court_ocr.watch import STATE_NAME, StateStore, Watcher         # noqa: E402


def _reconfigure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="court-ocr",
        description="Распознавание судебных приказов: мониторинг папки или разовый проход.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-i", "--input", metavar="DIR",
                   help="папка со сканами (перекрывает court-ocr.json)")
    p.add_argument("-o", "--output", metavar="DIR",
                   help="папка результатов (перекрывает court-ocr.json)")
    p.add_argument("-j", "--threads", type=int, metavar="N", help="число потоков")
    p.add_argument("--interval", type=float, metavar="SEC",
                   help="период опроса папки в режиме мониторинга, с")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--watch", dest="watch", action="store_true",
                      help="сразу запустить непрерывный мониторинг")
    mode.add_argument("--once", dest="watch", action="store_false",
                      help="разовый проход по папке")
    p.set_defaults(watch=None)
    p.add_argument("--no-menu", action="store_true",
                   help="не показывать меню — для ярлыка, автозагрузки, планировщика")
    p.add_argument("--rescan", action="store_true",
                   help="забыть обработанное и взять все файлы в папке заново")
    p.add_argument("--save-config", action="store_true",
                   help="сохранить переданные пути и потоки в court-ocr.json")
    return p.parse_args(argv)


def ask_threads(max_threads: int) -> int:
    """Спросить число потоков. По умолчанию — автоопределённый максимум (ядра CPU)."""
    limit = max(128, max_threads)
    print(f"Доступно потоков (ядер) — определено автоматически: {max_threads}.")
    while True:
        raw = input(f"Сколько потоков использовать? [Enter = максимум {max_threads}]: ").strip()
        if not raw:
            return max_threads
        try:
            n = int(raw)
            if 1 <= n <= limit:
                return n
        except ValueError:
            pass
        print(f"  Введите целое число от 1 до {limit}.")


def _fmt(sec: float) -> str:
    return f"{sec:.1f}с" if sec < 60 else f"{int(sec // 60)}м {int(sec % 60)}с"


def _mean_conf(rows) -> float:
    vals = [float(r["ocr_confidence"]) for r in rows
            if str(r.get("ocr_confidence", "")).replace(".", "", 1).isdigit()]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def process_batch(files, tess: Tesseract, cfg: Config, threads: int,
                  output_dir: Path) -> list:
    """Обработать файлы по документам: каждый документ сразу пишет свой PDF + CSV."""
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
        pages = info["pages"]
        sec = 0.0
        for r in pages:
            try:
                sec += float(r.get("seconds") or 0)
            except ValueError:
                pass
        parts = [info["csv"].name]
        if info["pdf"]:
            parts.append(info["pdf"].name)
        print(f"  ✓ Документ «{info['stem']}» готов: {len(pages)} стр., "
              f"{info['n_lines']} строк, машинное время {_fmt(sec)}, "
              f"точность {_mean_conf(pages)}%")
        print("      → в папке output: " + "  +  ".join(parts))
        print("-" * 64)

    docs = run_per_document(files, tess, cfg, threads, output_dir,
                            on_page=on_page, on_doc=on_doc)
    elapsed = time.perf_counter() - t0

    all_rows = [r for d in docs for r in d["pages"]]
    errs = sum(1 for r in all_rows if str(r.get("status", "")).startswith("error"))
    print("=" * 64)
    print(f"Готово за {_fmt(elapsed)}. Документов: {len(docs)}, страниц: {len(all_rows)} "
          f"(ошибок: {errs}), средняя точность {_mean_conf(all_rows)}%.")
    print(f"Результаты (CSV + PDF на каждый файл) в папке: {output_dir}")
    return docs


def run_once(watcher: Watcher, tess: Tesseract, cfg: Config, threads: int,
             output_dir: Path) -> None:
    """Разовый проход: взять то, что уже лежит в папке, и обработать."""
    files = watcher.poll()
    if not files:
        # Первый опрос только «знакомится» с файлами (проверка на дописанность),
        # поэтому для разового прохода делаем нужное число опросов подряд.
        for _ in range(watcher.stable_checks):
            time.sleep(min(0.5, watcher.interval))
            files = watcher.poll()
            if files:
                break
    if not files:
        print(f"\nНовых файлов нет. Папка входа: {watcher.paths[0]}")
        return
    try:
        process_batch(files, tess, cfg, threads, output_dir)
    finally:
        for f in files:
            watcher.mark_done(f)


def run_watch(watcher: Watcher, tess: Tesseract, cfg: Config, threads: int,
              output_dir: Path) -> None:
    """Непрерывный мониторинг: распознаём документы по мере появления."""
    print()
    print("=" * 64)
    print("  МОНИТОРИНГ ЗАПУЩЕН — новые файлы распознаются автоматически")
    print(f"  Слежу за папкой: {watcher.paths[0]}")
    print(f"  Опрос раз в {watcher.interval:g} с. Ctrl+C — остановить.")
    print("=" * 64)

    idle_drawn = False

    def handler(files):
        nonlocal idle_drawn
        if idle_drawn:
            print()  # закрыть строку ожидания, которая печаталась через \r
            idle_drawn = False
        process_batch(files, tess, cfg, threads, output_dir)

    def on_idle():
        nonlocal idle_drawn
        print(f"\r  {time.strftime('%H:%M:%S')}  ожидание новых файлов… "
              f"(обработано: {len(watcher.state)})   ", end="", flush=True)
        idle_drawn = True

    try:
        watcher.run(handler, on_idle=on_idle)
    except KeyboardInterrupt:
        if idle_drawn:
            print()
        print("\n  Мониторинг остановлен.")


def edit_settings(data: dict) -> dict:
    """Спросить новые пути/потоки/интервал и предложить сохранить их в JSON."""
    print("\n  Enter — оставить текущее значение.")

    def ask(prompt: str, current) -> str:
        raw = input(f"  {prompt} [{current}]: ").strip().strip('"')
        return raw or str(current)

    data = dict(data)
    data["input_dir"] = ask("Папка со сканами", data["input_dir"])
    data["output_dir"] = ask("Папка результатов", data["output_dir"])

    raw = ask("Потоков (0 — по числу ядер)", data["threads"])
    try:
        data["threads"] = max(0, int(raw))
    except ValueError:
        print("  Не число — оставляю прежнее значение потоков.")

    raw = ask("Опрос папки, секунд", f"{float(data['watch_interval']):g}")
    try:
        data["watch_interval"] = max(0.2, float(raw.replace(",", ".")))
    except ValueError:
        print("  Не число — оставляю прежний интервал.")

    ans = input("  Сохранить в court-ocr.json? [Enter — да, n — нет]: ").strip().lower()
    if ans not in ("n", "no", "н", "нет"):
        try:
            path = settings.save(BASE, data)
            print(f"  Сохранено: {path}")
        except OSError as exc:
            print(f"  [!] Не удалось сохранить настройки: {exc}")
    return data


def show_settings(input_dir: Path, output_dir: Path, threads: int,
                  interval: float, watch_default: bool) -> None:
    cfg_file = settings.config_path(BASE)
    print(f"  Папка со сканами:   {input_dir}")
    print(f"  Папка результатов:  {output_dir}")
    print(f"  Потоков:            {threads}")
    print(f"  Опрос папки:        раз в {interval:g} с")
    print(f"  Файл настроек:      {cfg_file}"
          f"{'' if cfg_file.is_file() else '  (нет — значения по умолчанию)'}")
    print(f"  Режим по умолчанию: {'мониторинг' if watch_default else 'разовый проход'}")


def _make_watcher(input_dir: Path, output_dir: Path, data: dict) -> Watcher:
    state = StateStore(output_dir / STATE_NAME)
    return Watcher(
        [input_dir],
        recursive=bool(data["recursive"]),
        interval=float(data["watch_interval"]),
        stable_checks=int(data["watch_stable"]),
        exclude=[output_dir],   # результаты не должны попадать обратно на вход
        state=state,
    )


def main(argv=None) -> int:
    _reconfigure_console()
    args = _parse_args(argv)

    data = settings.load(BASE)
    if args.input:
        data["input_dir"] = args.input
    if args.output:
        data["output_dir"] = args.output
    if args.threads is not None:
        data["threads"] = args.threads
    if args.interval is not None:
        data["watch_interval"] = args.interval
    if args.watch is not None:
        data["watch"] = args.watch
    if args.save_config:
        try:
            print(f"Настройки сохранены: {settings.save(BASE, data)}")
        except OSError as exc:
            print(f"[!] Не удалось сохранить настройки: {exc}")

    print("=" * 64)
    print("  Распознавание судебных приказов → CSV + PDF")
    print("=" * 64)

    # Движок Tesseract вшит в exe; языковая модель — во внешней папке tessdata/.
    tess_cmd = find_tesseract()
    if not tess_cmd:
        print("\nОШИБКА: не найден движок Tesseract.")
        if not args.no_menu:
            input("\nНажмите Enter для выхода...")
        return 2

    model_dir = BASE / "tessdata"
    tessdata = str(model_dir) if model_dir.is_dir() else find_tessdata(tess_cmd)
    if not tessdata or not (Path(tessdata) / "rus.traineddata").is_file():
        print("\nОШИБКА: не найдена языковая модель.")
        print(f"Положите rus.traineddata и osd.traineddata в папку:\n  {model_dir}")
        if not args.no_menu:
            input("\nНажмите Enter для выхода...")
        return 2

    tess = Tesseract(tess_cmd, tessdata=tessdata, lang="rus", psm=6, oem=1, dpi=300)

    while True:
        input_dir = settings.resolve_dir(BASE, data["input_dir"])
        output_dir = settings.resolve_dir(BASE, data["output_dir"])
        try:
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"\nОШИБКА: не удалось создать папку: {exc}")
            if args.no_menu:
                return 2
            data = edit_settings(data)
            continue

        threads = int(data["threads"]) or (os.cpu_count() or 4)
        cfg = Config(dpi=300, keep_raw=True, make_pdf=bool(data["make_pdf"]),
                     base_dir=input_dir)

        print(f"  Языковая модель:    {tessdata}")
        show_settings(input_dir, output_dir, threads, float(data["watch_interval"]),
                      bool(data["watch"]))

        watcher = _make_watcher(input_dir, output_dir, data)
        if args.rescan:
            watcher.state.clear()
            args.rescan = False  # только для первого запуска

        if args.no_menu:
            if data["watch"]:
                run_watch(watcher, tess, cfg, threads, output_dir)
            else:
                run_once(watcher, tess, cfg, threads, output_dir)
            return 0

        print()
        print("  [Enter] непрерывный мониторинг папки (Ctrl+C — стоп)")
        print("  [1] разовый проход по папке")
        print("  [2] изменить папки, потоки, интервал")
        print("  [3] забыть обработанное (взять все файлы заново)")
        print("  [q] выход")
        try:
            choice = input("  Выбор: ").strip().lower()
        except EOFError:
            return 0

        if choice in ("q", "quit", "exit", "выход", "в"):
            break
        if choice == "2":
            data = edit_settings(data)
            continue
        if choice == "3":
            watcher.state.clear()
            n = sum(1 for _ in iter_input_files([input_dir], bool(data["recursive"])))
            print(f"  Память очищена: при следующем проходе будет взято файлов: {n}")
            continue
        if choice == "1":
            run_once(watcher, tess, cfg, threads, output_dir)
        else:
            run_watch(watcher, tess, cfg, threads, output_dir)

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
