import abc
import copy
import re
from typing import Callable, Mapping, List, Tuple


def get_initial_form(word: str, lemmer_type: str = "empty") -> str:
    return lemmers[lemmer_type](word)


class BaseLemmer(abc.ABC):
    vowel = r"аеёиоуыэюя"
    regex_perfective_gerunds = [
        r"(в|вши|вшись)$",
        r"(ив|ивши|ившись|ыв|ывши|ывшись)$",
    ]
    old_signs = [
        r"(i|ѣ|ъ|ъ)$",
    ]
    regex_adjective = [
        r"(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$",
    ]
    regex_participle = [
        r"(ем|нн|вш|ющ|щ)",
        r"(ивш|ывш|ующ)",
    ]
    regex_reflexives = [
        r"(ся|сь)$",
    ]
    regex_verb = [
        r"(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)$",
        r"(ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)$",
    ]
    regex_noun = [
        r"(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|iи|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|iя|ья|я)$",
    ]
    regex_superlative = [
        r"(ейш|ейше)$",
    ]
    regex_derivational = [
        r"(ост|ость)$",
    ]
    regex_i = [
        r"и$",
    ]
    regex_nn = [
        r"нн$",
    ]
    regex_soft_sign = [
        r"ь$",
    ]

    def __init__(self) -> None:
        self.cache = {}

    def is_vowel(self, char: str) -> bool:
        return char in self.vowel

    def check_regexp(self, regexp: str, text: str) -> bool:
        return re.search(regexp, text) is not None

    @abc.abstractmethod
    def __call__(self, word) -> str:
        pass


class SmaltStemmer(BaseLemmer):
    def __init__(self):
        super().__init__()
        self.base_word = ""

    def __call__(self, word: str) -> str:
        if word in self.cache:
            return self.cache[word]

        rv, r2 = self.find_regions(word)
        self.base_word = copy.copy(word).lower()

        self.remove_endings(self.old_signs, rv)
        if not self.remove_endings(self.regex_perfective_gerunds, rv):
            self.remove_endings(self.regex_reflexives, rv)
            if not (
                self.remove_endings(
                    [
                        self.regex_participle[0] + self.regex_adjective[0],
                        self.regex_participle[1] + self.regex_adjective[0],
                    ],
                    rv,
                )
                or self.remove_endings(self.regex_adjective, rv)
            ):
                if not self.remove_endings(self.regex_verb, rv):
                    self.remove_endings(self.regex_noun, rv)
        self.remove_endings(self.regex_i, rv)
        self.remove_endings(self.regex_derivational, r2)
        if self.remove_endings(self.regex_nn, rv):
            self.base_word += "н"
        self.remove_endings(self.regex_superlative, rv)
        self.remove_endings(self.regex_soft_sign, rv)
        self.cache[word] = self.base_word

        return self.base_word

    def remove_endings(self, regexps: List[str], region: int) -> bool:
        prefix = self.base_word[0:region]
        suffix = self.base_word[region:]
        if len(regexps) == 2:
            if self.check_regexp(".+[а|я]" + regexps[0], suffix):
                self.base_word = prefix + re.sub(regexps[0], "", suffix)
                return True
        if self.check_regexp(".+" + regexps[-1], suffix):
            self.base_word = prefix + re.sub(regexps[-1], "", suffix)
            return True
        return False

    def find_regions(self, word: str) -> Tuple[int, int]:
        state = 0
        rv = 0
        for i in range(1, len(word)):
            prev_char = word[i - 1]
            char = word[i]
            if state == 0:
                if self.is_vowel(char):
                    rv = i + 1
                    state = 1
            elif state == 1:
                if self.is_vowel(prev_char) and not self.is_vowel(char):
                    state = 2
            elif state == 2:
                if self.is_vowel(prev_char) and not self.is_vowel(char):
                    return rv, i + 1
        return rv, 0


lemmers: Mapping[str, Callable[[str], str]] = {
    "empty": lambda w: " ",
    "smalt_stemmer": SmaltStemmer(),
}
