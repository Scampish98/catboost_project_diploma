import re
import string
from collections import defaultdict

import click
import nltk

from lib import (
    config as global_config,
    dataframe,
    lemmers,
    predict,
    recovery,
)

punctuation = f"{string.punctuation}«»—’"
re_punctuation = re.compile(f"^[{punctuation}]*$")


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    help="Path to config in yaml",
)
@click.option(
    "--meta",
    "-m",
    type=str,
    help="Path to meta in json",
)
@click.option("--input", type=str, help="Path to input text")
@click.option("--output", type=str, help="Path to output file")
def process(config: str, meta: str, input: str, output: str) -> None:
    config = global_config.Config.from_file(config)
    with open(output, mode="w", encoding="utf-8") as f:
        f.write(recovery.recovery_result(prepare_and_predict(config, meta, input)))


def prepare_and_predict(
    config: global_config.Config, meta: str, input: str
) -> dataframe.Dataframe:
    result = predict.predict(
        prepare_text(input, config),
        config=config,
        meta_path=meta,
    )
    result.sort(["chapter_id", "paragraph_id", "sentence_id", "word_id"])

    return result


def prepare_text(path: str, config: global_config.Config) -> dataframe.Dataframe:
    words_data = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    paragraphs = filter(None, text.split("\n"))
    for paragraph_id, paragraph_text in enumerate(paragraphs):
        for sentence_id, sentence_text in enumerate(
            nltk.tokenize.sent_tokenize(paragraph_text)
        ):
            words = [
                word
                for word in nltk.tokenize.word_tokenize(sentence_text)
                if re_punctuation.search(word) is None
            ]
            for word_id, word in enumerate(words):
                word = word.strip(punctuation).lower()
                words_data["WORD"].append(word)
                words_data["initial_form"].append(
                    lemmers.get_initial_form(word, config.lemmer_type)
                )
                words_data["text_id"].append("0")
                words_data["chapter_id"].append("0")
                words_data["word_id"].append(str(word_id))
                words_data["sentence_id"].append(str(sentence_id))
                words_data["paragraph_id"].append(str(paragraph_id))
                for shift in range(
                    -config.number_words_before, config.number_words_after + 1
                ):
                    words_data[f"word_{shift}"].append(
                        words[word_id + shift]
                        if 0 <= word_id + shift < len(words)
                        else "0"
                    )

    names, data = zip(*words_data.items())
    return dataframe.Dataframe(
        size=len(data[-1]) if data else 0,
        data=data,
        names=names,
        logging_level=config.logging_level,
    )


if __name__ == "__main__":
    process()
