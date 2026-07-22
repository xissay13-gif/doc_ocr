"""Извлечение полей судебного приказа из распознанного текста.

Все функции устойчивы к «шуму» OCR и к вариантам вёрстки (штрихкод сверху или
снизу, «ПРИКАЗЫВАЮ» либо «РЕШИЛ», разные форматы паспорта и сумм). Если поле не
найдено — возвращается пустая строка/None, исключения не выбрасываются: сырой
текст всегда сохраняется отдельной колонкой, поэтому данные не теряются.
"""
from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Optional

# Поля CSV в порядке колонок. pipeline использует этот список как заголовок.
FIELDS = [
    "file",
    "page",
    "doc_type",
    "barcode",
    "case_number",
    "date",
    "court",
    "court_number",
    "judge",
    "claimant",
    "claimant_inn",
    "debtors",
    "debtor_births",
    "debtor_passports",
    "debtor_address",
    "debt_subject",
    "debt_amount",
    "penalty",
    "state_fee",
    "period_from",
    "period_to",
    "rotation_deg",
    "ocr_confidence",
    "ocr_chars",
    "seconds",
    "status",
    "raw_text",
]

_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11,
    "декабря": 12,
    # варианты «мая», допускаемые регуляркой ма[йея] (в т.ч. огрехи OCR)
    "май": 5, "мае": 5,
}


# --------------------------------------------------------------------------- #
# Нормализация текста
# --------------------------------------------------------------------------- #
def _clean(text: str) -> str:
    """Нормализуем юникод и заменяем неразрывные пробелы обычными.

    Используем NFC, а не NFKC: NFKC раскладывает «№» (U+2116) в «No» и ломает
    поиск номеров участка и дела.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    return text


def _flat(text: str) -> str:
    """Схлопываем любые пробелы/переводы строк в одиночный пробел."""
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Разбор денежных сумм
# --------------------------------------------------------------------------- #
# Две формы записи:
#   «85 713,31 руб.»   -> рубли,копейки в одном числе
#   «6151 руб. 12 коп.» -> рубли и копейки раздельно
#   «14554,76 рублей»  -> тоже единое число
_AMOUNT_RE = re.compile(
    r"(?P<rub>\d[\d ]*)"                 # рублёвая часть (может содержать пробелы-разряды)
    r"(?:[.,](?P<kop1>\d{2}))?"          # копейки через запятую/точку
    r"\s*руб\w*\.?"                      # «руб», «руб.», «рублей»
    r"(?:\s*(?P<kop2>\d{1,2})\s*коп)?",  # либо «NN коп.»
    re.IGNORECASE,
)


def parse_amount(fragment: str) -> Optional[str]:
    """Первую денежную сумму во фрагменте возвращаем как строку «12345.67»."""
    if not fragment:
        return None
    m = _AMOUNT_RE.search(fragment)
    if not m:
        return None
    rub = re.sub(r"\D", "", m.group("rub"))
    if not rub:
        return None
    kop = m.group("kop1") or m.group("kop2") or "0"
    kop = kop.zfill(2)[:2]
    try:
        return str(Decimal(f"{int(rub)}.{kop}"))
    except (InvalidOperation, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Регулярные выражения для отдельных полей
# --------------------------------------------------------------------------- #
_BARCODE_RE = re.compile(r"([A-ZА-Я]\s?\d{4,6}\s?[-–—]\s?\d{2}\s?[-–—]\s?\d{8,12})")
_CASE_RE = re.compile(r"Дел[оа]\s*\S{0,2}\s*(\d[\d ]{0,3}[-–—]\d{2,6}\s*/\s*\d{4})", re.IGNORECASE)
_INN_RE = re.compile(r"ИНН[:\s]*?(\d{10,12})", re.IGNORECASE)
_DOCTYPE_RE = re.compile(
    r"(СУДЕБНЫЙ\s+ПРИКАЗ|ЗАОЧНОЕ\s+РЕШЕНИЕ|РЕШЕНИЕ|ОПРЕДЕЛЕНИЕ|ПОСТАНОВЛЕНИЕ)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(\d{1,2})\s+("
    r"январ\w*|феврал\w*|март\w*|апрел\w*|ма[йея]|июн\w*|июл\w*|"
    r"август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*"
    r")\s+(\d{4})",
    re.IGNORECASE,
)
_COURT_RE = re.compile(
    r"(Судебн\w+\s+участ\w+\s+№?\s*\d+[^\n]*)", re.IGNORECASE
)
# «№» Tesseract может распознать как №, No, N или Ne — допускаем все варианты.
_COURT_NUM_RE = re.compile(
    r"Судебн\w+\s+участ\w+\s*(?:№|No|Ne|N)?\s*(\d+)", re.IGNORECASE
)
_CLAIMANT_RE = re.compile(
    r"(Акционерн\w+\s+обществ\w+|Публичн\w+\s+акционерн\w+\s+обществ\w+|"
    r"Общества?\s+с\s+ограниченн\w+\s+ответственност\w+|ООО|ПАО|АО)\s*"
    r"[«\"“]([^»\"”\n]{2,80})[»\"”]",
    re.IGNORECASE,
)
# ФИО «привязано» к дате рождения — самый надёжный якорь для должников.
# Между ФИО и датой может стоять клаузула «паспорт: NNN,» (встречается в части
# приказов), поэтому допускаем её опционально.
# БЕЗ re.IGNORECASE: опираемся на заглавную первую букву, чтобы отличать ФИО от
# служебных слов («должника», «взыскать»). С IGNORECASE «[А-ЯЁ]» ловил и строчные.
_DEBTOR_RE = re.compile(
    r"((?:[А-ЯЁ][а-яё]+[-\s]+){1,3}[А-ЯЁ][а-яё]+)\s*,?\s*"
    r"(?:[Пп]аспорт[^А-ЯЁ]{0,40})?"
    r"(\d{2}\.\d{2}\.\d{4})\s*(?:года|г\.?)\s*рожд",
)
_PASSPORT_RE = re.compile(
    r"паспорт\w*[:\s]*(?:серия\s*)?(\d{2}\s?\d{2})\D{0,14}?(\d{6})", re.IGNORECASE
)
_ADDRESS_RE = re.compile(
    r"(?:прожива\w+\s+по\s+адресу|адрес\w*\s+прожива\w+|зарегистрирован\w*\s+по\s+адресу)"
    r"[:\s]*(.+?)(?=,?\s*(?:в\s+пользу|исследовав|задолженност|$))",
    re.IGNORECASE,
)
_PERIOD_RE = re.compile(
    r"за\s+период\s+с\s+(\d{2}\.\d{2}\.\d{4})\D*?по\s+(\d{2}\.\d{2}\.\d{4})",
    re.IGNORECASE,
)
_SUBJECT_RE = re.compile(
    r"задолженност\w*\s+(за\s+.+?)(?=\s+за\s+период|,\s*пени|\s+в\s+размере|$)",
    re.IGNORECASE,
)
# И.О. Фамилия  или  Фамилия И.О.
_JUDGE_INITIALS_FIRST = re.compile(r"([А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+)")
_JUDGE_SURNAME_FIRST = re.compile(r"([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.)")


def _first(regex: re.Pattern, text: str, group: int = 1) -> str:
    m = regex.search(text)
    return _flat(m.group(group)) if m else ""


def _norm_case(raw: str) -> str:
    return re.sub(r"\s+", "", raw) if raw else ""


def _dedupe(seq):
    """Убрать повторы, сохранив порядок (ФИО/паспорта дублируются: вводная + резолютивная часть)."""
    seen, out = set(), []
    for item in seq:
        key = item if isinstance(item, str) else tuple(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _parse_date(text: str) -> str:
    m = _DATE_RE.search(text)
    if not m:
        return ""
    day = int(m.group(1))
    month = _MONTHS.get(m.group(2).lower())
    # допускаем усечённые формы («декабр»), ищем по префиксу
    if month is None:
        for name, num in _MONTHS.items():
            if name.startswith(m.group(2).lower()[:5]):
                month = num
                break
    year = int(m.group(3))
    if month is None:
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def _extract_judge(text: str) -> str:
    """Ищем подпись судьи — берём совпадение ближе к концу документа."""
    candidates = list(_JUDGE_INITIALS_FIRST.finditer(text)) + list(
        _JUDGE_SURNAME_FIRST.finditer(text)
    )
    if not candidates:
        return ""
    best = max(candidates, key=lambda m: m.start())
    return _flat(best.group(1))


# --------------------------------------------------------------------------- #
# Главная функция
# --------------------------------------------------------------------------- #
def extract_fields(raw_text: str) -> dict:
    """Из распознанного текста собираем словарь полей судебного приказа."""
    text = _clean(raw_text)
    flat = _flat(text)

    row: dict = {k: "" for k in FIELDS}

    dt = _first(_DOCTYPE_RE, flat)
    row["doc_type"] = (dt[:1].upper() + dt[1:].lower()) if dt else ""
    row["barcode"] = re.sub(r"\s", "", _first(_BARCODE_RE, text))
    row["case_number"] = _norm_case(_first(_CASE_RE, flat))
    row["date"] = _parse_date(flat)
    row["court"] = _first(_COURT_RE, text)
    row["court_number"] = _first(_COURT_NUM_RE, flat)
    row["judge"] = _extract_judge(text)

    m_claim = _CLAIMANT_RE.search(flat)
    if m_claim:
        form = _flat(m_claim.group(1))
        name = _flat(m_claim.group(2))
        row["claimant"] = f"{form} «{name}»"
    row["claimant_inn"] = _first(_INN_RE, flat)

    # Должники: пары «ФИО + дата рождения» (ФИО и дата дублируются во вводной и
    # резолютивной частях — дедуплицируем пары вместе, чтобы не рассинхронить списки).
    pairs = _dedupe([(_flat(m.group(1)), m.group(2)) for m in _DEBTOR_RE.finditer(flat)])
    row["debtors"] = "; ".join(name for name, _ in pairs)
    row["debtor_births"] = "; ".join(birth for _, birth in pairs)

    # Паспорта (серия + номер) — отдельным списком, без повторов.
    passports = _dedupe(
        [f"{s.replace(' ', '')} {n}" for s, n in _PASSPORT_RE.findall(flat)]
    )
    row["debtor_passports"] = "; ".join(passports)

    row["debtor_address"] = _first(_ADDRESS_RE, flat)
    row["debt_subject"] = _first(_SUBJECT_RE, flat)

    m_period = _PERIOD_RE.search(flat)
    if m_period:
        row["period_from"] = m_period.group(1)
        row["period_to"] = m_period.group(2)

    # Суммы. В резолютивной части суд всегда указывает их по порядку:
    #   основной долг → пени → госпошлина.
    # Поэтому: долг = первая сумма после «Взыскать»; пени = первая сумма после
    # слова «пени», не совпадающая с суммой долга; госпошлина = первая сумма
    # после «пошлин». Такой подход устойчив к формулировкам вроде «включая пени,
    # в размере X» (одна сумма) и к слову «степени» (используем границу слова).
    low = flat.lower()
    op = low.find("взыскать")
    if op < 0:
        op = 0
    region, rlow = flat[op:], low[op:]
    amounts = list(_AMOUNT_RE.finditer(region))

    debt_pos = None
    if amounts:
        row["debt_amount"] = parse_amount(amounts[0].group(0)) or ""
        debt_pos = amounts[0].start()

    gp = rlow.rfind("пошлин")

    mp = re.search(r"\bпени\b", rlow)
    if mp:
        for m in amounts:
            # первая сумма после «пени», но до госпошлины и не равная сумме долга
            # (иначе при формулировке «включая пени, в размере X» пени = долгу
            # или ошибочно берётся сумма госпошлины)
            if (m.start() >= mp.start() and m.start() != debt_pos
                    and (gp < 0 or m.start() < gp)):
                row["penalty"] = parse_amount(m.group(0)) or ""
                break

    if gp >= 0:
        for m in amounts:
            if m.start() >= gp:
                row["state_fee"] = parse_amount(m.group(0)) or ""
                break

    return row
