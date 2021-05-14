import csv
import os
import re
import time
from textblob import TextBlob
from typing import List, Mapping
from urllib.error import HTTPError

from .logging import LoggedClass

LANGUAGES = os.path.join("..", "..", "data/languages.tsv")


class LanguageDetector(LoggedClass):
    russian = re.compile(r"[а-яА-Я]")
    old_signs = re.compile(r"(ѣ|ъ|ъ)")
    languages: Mapping[str, List[str]] = {
        "ru": ["Русский", "0"],
        "it": ["Итальянский", "389"],
        "fr": ["Французский", "385"],
        "en": ["Английский", "386"],
        "la": ["Латинский", "388"],
        "de": ["Немецкий", "387"],
        "pl": ["Польский", "390"],
        "el": ["Греческий", "391"],
    }
    last_queries = []

    def check_russian(self, word: str) -> bool:
        if self.old_signs.search(word) is not None:
            return True
        russian_lettres = 0
        for char in word:
            russian_lettres += 1 if self.russian.search(char) is not None else 0
        return russian_lettres * 2 >= len(word)

    def get_language_iso_batch(self, words: List[str]) -> List[str]:
        result = []
        cache = {}
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), LANGUAGES),
            "r",
            encoding="utf-8",
        ) as input_stream:
            tsv_reader = csv.reader(input_stream, delimiter="\t", lineterminator="\n")
            for row in tsv_reader:
                cache[row[0]] = row[1]

        was_added = False
        for word in words:
            if word not in cache:
                iso = self.get_language_iso(word)
                cache[word] = iso
                was_added = True
            result.append(cache[word])

        if was_added:
            with open(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), LANGUAGES),
                "w",
                newline="",
                encoding="utf-8",
            ) as output_stream:
                writer = csv.writer(output_stream, delimiter="\t")
                for key, value in cache.items():
                    writer.writerow([key, value])
        return result

    def get_language_iso(self, word: str) -> str:
        self._logger.debug("%s get_language_iso", word)
        if self.check_russian(word):
            return "ru"
        languages = set()
        for _ in range(5):
            try:
                while len(self.last_queries) == 500:
                    if time.time() - self.last_queries[0] > 60:
                        self.last_queries.pop(0)
                    else:
                        time.sleep(0.1)
                lang = TextBlob(word).detect_language()
                self.last_queries.append(time.time())
                if "un" != lang:
                    languages.add(lang)
            except HTTPError as e:
                self._logger.info("Http Error during detect language: %s", e)
                time.sleep(1)
                continue
            except Exception as e:
                self._logger.debug("Exception during detect language: %s", e)
            break

        if "ru" in languages:
            return "ru"
        if len(languages) == 1:
            return list(languages)[0]
        self._logger.debug("Unknown language: %s", word)
        return "unknown"

    def get_language_batch(self, words: List[str]) -> List[List[str]]:
        result = []
        for iso in self.get_language_iso_batch(words):
            result.append(self.languages.get(iso, ["Другой", "392"]))
        return result

    def get_language(self, word: str) -> List[str]:
        return self.languages.get(self.get_language_iso(word), ["Другой", "392"])
