"""CSV и распознанный PDF можно класть в разные папки.

Tesseract не нужен: write_document получает уже готовые страницы и байты PDF.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from court_ocr.pipeline import write_document  # noqa: E402
from court_ocr.settings import resolve_optional_dir  # noqa: E402


def one_page_pdf() -> bytes:
    """Настоящий одностраничный PDF — write_document склеивает страницы через MuPDF."""
    import fitz

    doc = fitz.open()
    doc.new_page(width=200, height=280)
    data = doc.tobytes()
    doc.close()
    return data


PAGE_MAP = {0: ({"page": 1, "lines": ["СУДЕБНЫЙ ПРИКАЗ", "Дело № 2-1/2024"]}, None)}


class OutputDirsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.pdf_bytes = one_page_pdf()
        self.pages = {0: (PAGE_MAP[0][0], self.pdf_bytes)}

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_split_dirs(self):
        csv_dir, pdf_dir = self.root / "csv", self.root / "pdf"
        info = write_document(Path("приказ.pdf"), self.pages, csv_dir, pdf_dir)

        self.assertEqual(info["csv"], csv_dir / "приказ.csv")
        self.assertEqual(info["pdf"], pdf_dir / "приказ.pdf")
        self.assertTrue(info["csv"].is_file())
        self.assertTrue(info["pdf"].is_file())
        # В чужой папке ничего лишнего не появилось.
        self.assertEqual([p.name for p in csv_dir.iterdir()], ["приказ.csv"])
        self.assertEqual([p.name for p in pdf_dir.iterdir()], ["приказ.pdf"])

    def test_same_dir_when_pdf_dir_omitted(self):
        out = self.root / "out"
        info = write_document(Path("приказ.pdf"), self.pages, out)
        self.assertEqual(info["csv"].parent, out)
        self.assertEqual(info["pdf"].parent, out)
        self.assertEqual(info["csv_dir"], info["pdf_dir"])

    def test_dirs_created_on_demand(self):
        csv_dir = self.root / "нет" / "такой" / "папки"
        pdf_dir = self.root / "и" / "такой" / "тоже"
        write_document(Path("приказ.pdf"), self.pages, csv_dir, pdf_dir)
        self.assertTrue((csv_dir / "приказ.csv").is_file())
        self.assertTrue((pdf_dir / "приказ.pdf").is_file())

    def test_csv_content_unaffected_by_split(self):
        info = write_document(Path("приказ.pdf"), self.pages,
                              self.root / "csv", self.root / "pdf")
        text = info["csv"].read_text(encoding="cp1251")
        self.assertIn("page;line_id;line", text)
        self.assertIn("1;1;СУДЕБНЫЙ ПРИКАЗ", text)

    def test_no_pdf_written_when_disabled(self):
        csv_dir, pdf_dir = self.root / "csv", self.root / "pdf"
        info = write_document(Path("приказ.pdf"), self.pages, csv_dir, pdf_dir,
                              make_pdf=False)
        self.assertIsNone(info["pdf"])
        self.assertFalse(pdf_dir.exists())


class DescribeTargetsTest(unittest.TestCase):
    """Шапка мониторинга. Строится из кусков, а вызывается только в живом
    запуске — ровно здесь однажды и потерялся «+» между строками."""

    def setUp(self) -> None:
        import ocr_court
        self.describe = ocr_court.describe_targets
        self.out, self.csv, self.pdf = (Path("/o"), Path("/c"), Path("/p"))

    def test_single_dir(self):
        text = self.describe(self.out, self.out, self.out, True)
        self.assertIn("Результат: /o", text)
        self.assertIn("CSV + PDF", text)
        self.assertTrue(text.endswith("\n"))

    def test_single_dir_without_pdf(self):
        self.assertNotIn("PDF", self.describe(self.out, self.out, self.out, False))

    def test_split_dirs_show_both(self):
        text = self.describe(self.out, self.csv, self.pdf, True)
        self.assertIn("CSV в:     /c", text)
        self.assertIn("PDF в:     /p", text)

    def test_split_dirs_without_pdf(self):
        text = self.describe(self.out, self.csv, self.pdf, False)
        self.assertIn("/c", text)
        self.assertNotIn("PDF", text)


class OptionalDirTest(unittest.TestCase):
    """Пустое значение в настройках означает «как папка результатов», а не путь."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_blank_is_none(self):
        for blank in ("", "   ", None, '""'):
            with self.subTest(value=blank):
                self.assertIsNone(resolve_optional_dir(self.base, blank))

    def test_relative_and_absolute(self):
        self.assertEqual(resolve_optional_dir(self.base, "csv"), self.base / "csv")
        target = self.base / "где-то"
        self.assertEqual(resolve_optional_dir(self.base, str(target)), target)


if __name__ == "__main__":
    unittest.main()
