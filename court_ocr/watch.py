"""Непрерывное наблюдение за входной папкой: документы распознаются сразу.

Внешних зависимостей нет — используется периодический опрос (polling). Он
одинаково работает на сетевых дисках и в общих папках, где событийные API
(inotify / ReadDirectoryChangesW) часто не срабатывают или теряют события.

Три вещи, ради которых нужен этот модуль:

* **Файл берётся в работу только когда он «устоялся».** Пока Проводник копирует
  PDF, файл уже виден в папке, но дописан лишь наполовину — OCR получил бы
  обрезанный документ. Ждём, пока размер и время изменения не перестанут
  меняться несколько опросов подряд, и пока файл не начнёт открываться на
  чтение (в Windows копируемый файл заблокирован пишущим процессом).
* **Повторно один и тот же файл не обрабатывается.** Список обработанного можно
  хранить на диске (:class:`StateStore`), чтобы перезапуск программы не погнал
  всю папку заново. Файл с тем же именем, но изменившимся размером/временем,
  считается новым.
* **Папка результатов исключается из обхода.** Иначе распознанный PDF, если
  output лежит внутри input, тут же попал бы на вход снова — и так по кругу.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .render import iter_input_files

# Отпечаток файла: путь, размер, время изменения (целые секунды).
FileKey = Tuple[str, int, int]

DEFAULT_INTERVAL = 2.0   # период опроса папки, с
DEFAULT_STABLE = 2       # столько опросов подряд файл должен быть неизменным
STATE_NAME = ".court-ocr-state.json"


def file_key(path: Path) -> Optional[FileKey]:
    """Отпечаток файла. None — файл исчез или недоступен."""
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return (str(path), st.st_size, int(st.st_mtime))


def is_complete(path: Path) -> bool:
    """Файл дописан и не заблокирован пишущим процессом (проверка для Windows)."""
    try:
        with open(path, "rb") as f:
            f.read(1)
    except OSError:
        return False
    return True


class StateStore:
    """Память об уже обработанных файлах; при указанном пути — сохраняется на диск."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._keys: Set[FileKey] = set()
        if self.path and self.path.is_file():
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # повреждённое состояние — просто начинаем с чистого листа
        for item in data.get("done", []):
            if isinstance(item, (list, tuple)) and len(item) == 3:
                try:
                    self._keys.add((str(item[0]), int(item[1]), int(item[2])))
                except (TypeError, ValueError):
                    continue

    def save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"done": [list(k) for k in sorted(self._keys)]}
            self.path.write_text(json.dumps(payload, ensure_ascii=False),
                                 encoding="utf-8")
        except OSError:
            pass  # состояние вспомогательное: не сохранилось — работаем дальше

    def add(self, key: FileKey, save: bool = True) -> None:
        self._keys.add(key)
        if save:
            self.save()

    def clear(self) -> None:
        """Забыть обработанное — следующий проход возьмёт все файлы заново."""
        self._keys.clear()
        if self.path:
            try:
                self.path.unlink()
            except OSError:
                pass

    def __contains__(self, key: object) -> bool:
        return key in self._keys

    def __len__(self) -> int:
        return len(self._keys)


@dataclass
class Watcher:
    """Опрашивает папки и отдаёт файлы, готовые к распознаванию.

    paths          — папки (или отдельные файлы), за которыми следим;
    recursive      — заходить ли в подпапки;
    interval       — период опроса, с (используется в :meth:`run`);
    stable_checks  — сколько опросов подряд файл должен быть неизменным;
    exclude        — папки, которые не обходим (обычно — папка результатов);
    state          — память об обработанном.
    """

    paths: Sequence[Path]
    recursive: bool = True
    interval: float = DEFAULT_INTERVAL
    stable_checks: int = DEFAULT_STABLE
    exclude: Sequence[Path] = ()
    state: StateStore = field(default_factory=StateStore)

    def __post_init__(self) -> None:
        self.paths = [Path(p) for p in self.paths]
        self.exclude = [self._resolve(Path(p)) for p in self.exclude]
        self.stable_checks = max(1, int(self.stable_checks))
        self._pending: Dict[str, Tuple[FileKey, int]] = {}
        self._ready: Dict[str, FileKey] = {}

    # -- внутреннее ------------------------------------------------------- #
    @staticmethod
    def _resolve(path: Path) -> Path:
        try:
            return path.resolve()
        except OSError:
            return path.absolute()

    def _excluded(self, path: Path) -> bool:
        if not self.exclude:
            return False
        target = self._resolve(path)
        for d in self.exclude:
            if target == d or target.is_relative_to(d):
                return True
        return False

    # -- опрос ------------------------------------------------------------ #
    def poll(self) -> List[Path]:
        """Один опрос: вернуть файлы, которые устоялись и ещё не обрабатывались."""
        ready: List[Path] = []
        seen: Set[str] = set()

        for f in iter_input_files(self.paths, self.recursive):
            if self._excluded(f):
                continue
            key = file_key(f)
            if key is None or key[1] == 0:
                continue  # исчез или ещё нулевой длины (файл только создан)
            name = str(f)
            seen.add(name)
            if key in self.state:
                continue

            prev = self._pending.get(name)
            # Файл изменился между опросами → отсчёт стабильности с начала.
            count = prev[1] + 1 if prev is not None and prev[0] == key else 1
            self._pending[name] = (key, count)

            if count > self.stable_checks and is_complete(f):
                self._pending.pop(name, None)
                self._ready[name] = key
                ready.append(f)

        for name in [n for n in self._pending if n not in seen]:
            self._pending.pop(name, None)  # файл унесли до готовности
        return ready

    def mark_done(self, path: Path) -> None:
        """Запомнить файл как обработанный (вызывается после OCR)."""
        name = str(path)
        key = self._ready.pop(name, None) or file_key(Path(path))
        if key:
            self.state.add(key)

    def skip_existing(self) -> int:
        """Пометить всё, что уже лежит в папке, обработанным («только новые»)."""
        n = 0
        for f in iter_input_files(self.paths, self.recursive):
            if self._excluded(f):
                continue
            key = file_key(f)
            if key and key not in self.state:
                self.state.add(key, save=False)
                n += 1
        self.state.save()
        return n

    # -- цикл --------------------------------------------------------------- #
    def run(self,
            handler: Callable[[List[Path]], None],
            on_idle: Optional[Callable[[], None]] = None,
            stop: Optional[Callable[[], bool]] = None) -> None:
        """Бесконечный цикл: как только файлы готовы — отдать их handler.

        handler получает список путей; после его возврата (даже при ошибке) файлы
        помечаются обработанными, чтобы битый файл не крутился в цикле вечно.
        on_idle вызывается, когда новых файлов нет. stop() → True — выход.
        """
        while not (stop is not None and stop()):
            batch = self.poll()
            if batch:
                try:
                    handler(batch)
                finally:
                    for f in batch:
                        self.mark_done(f)
                continue  # пока шёл OCR, могли докинуть ещё — проверяем сразу
            if on_idle is not None:
                on_idle()
            time.sleep(self.interval)
