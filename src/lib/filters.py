from .language_detector import LanguageDetector
from typing import Optional, List


def filter_language(
    row: List[str],
    names: Optional[List[str]],
    logging_level: str = "stats",
) -> bool:
    if not names or "WORD" not in names:
        return True
    detector = LanguageDetector(logging_level=logging_level)
    word = row[names.index("WORD")]
    return detector.get_language_iso(word) == "ru"
