#!/usr/bin/env python3
"""Встраивание Tesseract в проект (папка ./tesseract рядом со скриптом).

Что делает:
  1) скачивает официальную Windows-сборку Tesseract 5.x (UB-Mannheim, Apache-2.0);
  2) распаковывает её «тихой» установкой во временную папку и копирует в ./tesseract
     (система при этом не трогается, права администратора не нужны);
  3) докачивает языковые модели rus / osd / eng (tessdata_fast) в ./tesseract/tessdata.

После этого ocr_court.py находит встроенный tesseract.exe автоматически.

Примеры:
  python setup_tesseract.py                 # движок + rus,osd,eng
  python setup_tesseract.py --data-only     # только языковые модели (движок уже есть)
  python setup_tesseract.py --best          # модели из tessdata_best (точнее, крупнее)
  python setup_tesseract.py --langs rus,eng,osd,kaz

Требуется только выход в интернет и ~70 МБ на движок + ~15 МБ на модели.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# Официальная сборка UB-Mannheim. Основной адрес — GitHub-releases (проверенный,
# стабильный HTTPS); зеркало — сайт университета Мангейма.
INSTALLER_URL = (
    "https://github.com/UB-Mannheim/tesseract/releases/download/"
    "v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe"
)
INSTALLER_MIRROR = (
    "https://digi.bib.uni-mannheim.de/tesseract/"
    "tesseract-ocr-w64-setup-5.5.0.20241111.exe"
)
TESSDATA_FAST = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/{}.traineddata"
TESSDATA_BEST = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/{}.traineddata"

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "tesseract"
UA = {"User-Agent": "court-ocr-setup/1.0"}


def _reconfigure_console() -> None:
    """Печатать по-русски даже когда вывод перенаправлен в файл или конвейер.

    В Windows у перенаправленного stdout кодировка берётся из локали (cp1252 и
    подобные), и первый же русский print падает с UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _download(url: str, dest: Path) -> None:
    """Скачать url → dest с показом прогресса."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = 1024 * 64
        while True:
            data = resp.read(chunk)
            if not data:
                break
            out.write(data)
            read += len(data)
            if total:
                pct = read * 100 // total
                sys.stdout.write(f"\r    {read // 1024 // 1024} / {total // 1024 // 1024} МБ ({pct}%)")
                sys.stdout.flush()
    sys.stdout.write("\r" + " " * 50 + "\r")
    print(f"    сохранено: {dest.name} ({dest.stat().st_size // 1024} КБ)")


def _find_installed(preferred: Path):
    """Найти tesseract.exe после установки: инсталлятор UB-Mannheim обычно
    игнорирует /D= и ставит движок в стандартное место (Program Files при правах
    администратора, иначе — LocalAppData). Берём самую свежую установку."""
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    pf = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    pf86 = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    candidates = [
        preferred,
        pf / "Tesseract-OCR",
        pf86 / "Tesseract-OCR",
        local / "Programs" / "Tesseract-OCR",
        local / "Tesseract-OCR",
    ]
    found = [c for c in candidates if c and (c / "tesseract.exe").is_file()]
    if not found:
        return None
    return max(found, key=lambda c: (c / "tesseract.exe").stat().st_mtime)


def install_engine(url: str, force: bool) -> None:
    """Скачать инсталлятор, «тихо» его выполнить и скопировать движок в ./tesseract."""
    exe = TARGET / ("tesseract.exe" if sys.platform == "win32" else "tesseract")
    if exe.is_file() and not force:
        print(f"Движок уже встроен: {exe} (для перезаписи добавьте --force)")
        return
    if sys.platform != "win32":
        print("Автоустановка движка реализована для Windows. На Linux/macOS установите "
              "tesseract пакетным менеджером, затем запустите с --data-only.")
        return

    tmpdir = Path(tempfile.gettempdir())
    installer = tmpdir / "court_ocr_tess_setup.exe"
    build_dir = tmpdir / "court_ocr_tess_build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    try:
        print("Скачиваю движок Tesseract...")
        try:
            _download(url, installer)
        except Exception as exc:
            print(f"  не удалось скачать с основного адреса ({exc}); пробую зеркало...")
            _download(INSTALLER_MIRROR, installer)

        print("Устанавливаю (тихий режим)...")
        # NSIS: /S — тихо, /D= — предпочтительная папка (последним аргументом, без
        # кавычек). Инсталлятор может её проигнорировать — фактическое место найдём ниже.
        subprocess.run(f'"{installer}" /S /D={build_dir}', shell=True, check=False)

        src = _find_installed(build_dir)
        if src is None:
            raise RuntimeError(
                "после установки не найден tesseract.exe. Возможные причины: установка "
                "отменена запросом UAC, либо нужны права администратора. Установите "
                "Tesseract вручную и запустите setup_tesseract.py --data-only."
            )

        print(f"Копирую движок: {src} → {TARGET}")
        # Не затираем уже скачанные модели: копируем всё, кроме tessdata.
        TARGET.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.name.lower() == "tessdata" and (TARGET / "tessdata").exists():
                # переносим только отсутствующие модели из установки
                for td in item.glob("*.traineddata"):
                    dst = TARGET / "tessdata" / td.name
                    if not dst.is_file():
                        _copy(td, dst)
                continue
            _copy(item, TARGET / item.name)
        print("Движок встроен.")
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
        try:
            installer.unlink()
        except OSError:
            pass


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def install_langs(langs, best: bool, force: bool) -> None:
    """Докачать языковые модели в ./tesseract/tessdata."""
    tessdata = TARGET / "tessdata"
    tessdata.mkdir(parents=True, exist_ok=True)
    tmpl = TESSDATA_BEST if best else TESSDATA_FAST
    print(f"Языковые модели ({'best' if best else 'fast'}): {', '.join(langs)}")
    for lang in langs:
        dest = tessdata / f"{lang}.traineddata"
        if dest.is_file() and not force:
            print(f"  = {lang}: уже есть")
            continue
        try:
            _download(tmpl.format(lang), dest)
        except Exception as exc:
            print(f"  ! {lang}: не удалось скачать ({exc})")


def main(argv=None) -> int:
    _reconfigure_console()   # до parse_args: справка и ошибки тоже по-русски
    p = argparse.ArgumentParser(description="Встроить Tesseract и языковые модели в проект.")
    p.add_argument("--data-only", action="store_true",
                   help="только скачать языковые модели (движок уже установлен/встроен)")
    p.add_argument("--langs", default="rus,osd",
                   help="языки через запятую (по умолчанию rus,osd; osd нужен для авто-поворота)")
    p.add_argument("--best", action="store_true",
                   help="брать модели из tessdata_best (точнее, но крупнее)")
    p.add_argument("--url", default=INSTALLER_URL, help="свой адрес инсталлятора Tesseract")
    p.add_argument("--force", action="store_true", help="перекачать/переустановить, даже если есть")
    args = p.parse_args(argv)

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if "osd" not in langs:
        langs.append("osd")  # osd нужен для авто-поворота

    print("=" * 60)
    print("Встраивание Tesseract в court-ocr")
    print("=" * 60)

    if not args.data_only:
        install_engine(args.url, args.force)
    install_langs(langs, args.best, args.force)

    exe = TARGET / ("tesseract.exe" if sys.platform == "win32" else "tesseract")
    print("-" * 60)
    if exe.is_file():
        print(f"Готово. Встроенный Tesseract: {exe}")
        print("Теперь можно запускать:  python ocr_court.py <папка_или_файл>")
    else:
        print("Движок не встроен (использованы только языковые модели).")
        print("Укажите путь к tesseract.exe флагом --tesseract при запуске ocr_court.py,")
        print("либо запустите setup_tesseract.py без --data-only на Windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
