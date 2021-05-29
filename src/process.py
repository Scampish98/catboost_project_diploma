from collections import defaultdict

import click
import nltk
from catboost.text_processing import Tokenizer

from lib import (
    attribute,
    config as global_config,
    dataframe,
    lemmers,
    predict,
    recovery,
)


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
        f.write(
            recovery.recovery_result(
                predict.predict(
                    prepare_text(input, config),
                    config=config,
                    meta_path=meta,
                )
            )
        )


def prepare_text(path: str, config: global_config.Config) -> dataframe.Dataframe:
    words_data = defaultdict(list)
    tokenizer = Tokenizer(
        lowercasing=True,
        separator_type="BySense",
        token_types=[
            "Word",
            "Number",
        ],
    )

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    paragraphs = filter(None, text.split("\n"))
    for paragraph_id, paragraph_text in enumerate(paragraphs):
        sentences = nltk.tokenize.sent_tokenize(paragraph_text)
        for sentence_id, sentence_text in enumerate(sentences):
            words = tokenizer.tokenize(sentence_text)
            for word_id, word in enumerate(words):
                print(word)
                words_data["WORD"].append(word)
                words_data["initial_form"].append(
                    lemmers.get_initial_form(word, config.lemmer_type)
                )
                words_data["word_id"].append(word_id)
                words_data["sentence_id"].append(sentence_id)
                words_data["paragraph_id"].append(paragraph_id)
                for shift in range(1, 6):
                    words_data[f"word_{-shift}"].append(
                        words[word_id - shift] if word_id - shift >= 0 else " "
                    )
                    words_data[f"word_{shift}"].append(
                        words[word_id + shift] if word_id + shift < len(words) else " "
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
