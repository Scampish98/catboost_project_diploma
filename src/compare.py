from __future__ import print_function
import sys

import io
import itertools
import numpy
import os
from contextlib import contextmanager
from typing import List, Mapping, Tuple

import click

from lib import (
    attribute,
    config as global_config,
    dataframe,
    models,
)

NON_COMPARED_NAMES = set(
    itertools.chain(
        {
            "WORD",
            "initial_form",
            "chapter_id",
            "word_id",
            "sentence_id",
            "paragraph_id",
            "text_id",
            "true_part_of_speech",
        },
        [f"word_{word_id}" for word_id in range(1, 21)],
        [f"word_{-word_id}" for word_id in range(1, 21)],
    )
)


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    help="Path to config in yaml",
    default="",
)
@click.option(
    "--meta",
    "-m",
    type=str,
    help="Path to meta in json",
    default="",
)
@click.option(
    "--result-data-directory", "-d", type=str, help="Path to result data tsv", default=""
)
@click.option(
    "--result-data-filename", "-f", type=str, help="Path to result data tsv", default=""
)
@click.option("--validate-data", "-v", type=str, help="Path to validate data tsv")
@click.option(
    "--save-output-to-model-directory",
    is_flag=True,
    help="Redirect output to a file in the model directory",
)
def compare(
    config: str,
    meta: str,
    result_data_directory: str,
    result_data_filename: str,
    validate_data: str,
    save_output_to_model_directory: bool,
) -> None:
    model_id = get_model_id(config, meta)
    result_data = os.path.join(
        result_data_directory or os.path.join("..", "models", model_id),
        result_data_filename or "predict.tsv"
    )

    result_data = dataframe.Dataframe.from_tsv(
        result_data
    ).get_sorted_by_name("WORD")
    validate_data = dataframe.Dataframe.from_tsv(
        validate_data
    ).get_sorted_by_name("WORD")

    total_equals = 0
    result_iterator = result_data.read()
    validate_iterator = validate_data.read()
    keys = [
        result_data.name_ids[key]
        for key in ["WORD", "paragraph_id", "sentence_id", "word_id"]
    ]
    while True:
        try:
            result_row = next(result_iterator)
            validate_row = next(validate_iterator)
            total_equals += (
                result_row[result_data.name_ids["WORD"]].lower()
                == validate_row[validate_data.name_ids["WORD"]].lower()
            )
        except StopIteration:
            break

    result_data = result_data[
        list(set(result_data.names).difference(NON_COMPARED_NAMES))
    ]

    validate_data = validate_data[result_data.names]

    with open_output(save_output_to_model_directory, model_id) as output_file:
        accuracy(result_data, validate_data, output_file)
        completeness(result_data, output_file)


@contextmanager
def open_output(
    save_output_to_model_directory: bool, model_id: str
) -> io.TextIOWrapper:
    if not save_output_to_model_directory:
        yield sys.stdout
    else:
        with open(os.path.join("..", "models", model_id, "compare.txt"), "w") as f:
            yield f


def get_model_id(config: str, meta: str) -> str:
    config = global_config.Config.from_file(config)
    model = models.create_model(config, meta)
    return model.model_id


def accuracy(
    result_data: dataframe.Dataframe,
    validate_data: dataframe.Dataframe,
    output_file: io.TextIOWrapper,
) -> None:
    print("Accuracy assessment", file=output_file)
    attribute_accuracy(result_data, validate_data, output_file)
    word_accuracy(result_data, validate_data, output_file)


def attribute_accuracy(
    result_data: dataframe.Dataframe,
    validate_data: dataframe.Dataframe,
    output_file: io.TextIOWrapper,
) -> None:
    print("\nAttribute accuracy assessment", file=output_file)

    total_cnt = 0
    total_bad_cnt = 0
    total_good_cnt = 0
    for name in result_data.names:
        result_column = result_data[name]
        validate_column = validate_data[name]
        current_total = min(len(result_column), len(validate_column))
        current_bad = 0
        current_good = 0
        for i in range(current_total):
            if validate_column[i] == "0":
                current_total -= 1
                continue
            if validate_column[i] != result_column[i]:
                current_bad += 1
            if validate_column[i] == result_column[i]:
                current_good += 1
        total_cnt += current_total
        total_bad_cnt += current_bad
        total_good_cnt += current_good
        if current_total != 0:
            print(
                f"Name = {name},"
                f" bad = {current_bad},"
                f" good = {current_good},"
                f" cnt = {current_total},"
                f" bad percent = {100.0 * current_bad / current_total}"
                f" good percent = {100.0 * current_good / current_total}",
                file=output_file,
            )
    print(
        f"Total bad = {total_bad_cnt},"
        f" total good = {total_good_cnt},"
        f" total cnt = {total_cnt},"
        f" total bad percent = {100.0 * total_bad_cnt / total_cnt}"
        f" total good percent = {100.0 * total_good_cnt / total_cnt}",
        file=output_file,
    )


def word_accuracy(
    result_data: dataframe.Dataframe,
    validate_data: dataframe.Dataframe,
    output_file: io.TextIOWrapper,
) -> None:
    print("\nWord accuracy assessment", file=output_file)

    correct_part_of_speech = 0
    unknown_part_of_speech = 0
    result_accuracy = []
    result_iterator = result_data.read()
    validate_iterator = validate_data.read()
    while True:
        try:
            result_row = next(result_iterator)
            validate_row = next(validate_iterator)
            accuracy = 0
            total = 0
            for attribute_name in result_data.names:
                validate = validate_row[validate_data.name_ids[attribute_name]]
                if validate == "0":
                    continue
                result = result_row[result_data.name_ids[attribute_name]]
                accuracy += result == validate
                total += 1
            if total:
                result_accuracy.append(accuracy / total * 100)

            if validate_row[validate_data.name_ids["1"]] != "0":
                correct_part_of_speech += (
                    result_row[result_data.name_ids["1"]]
                    == validate_row[validate_data.name_ids["1"]]
                )
            else:
                unknown_part_of_speech += 1
        except StopIteration:
            break

    percentiles = [5, 10, 15, 20, 25, 75, 90, 95]
    print(f"Mean: {numpy.mean(result_accuracy)}", file=output_file)
    print(f"Median: {numpy.median(result_accuracy)}", file=output_file)
    for index, percentile in enumerate(numpy.percentile(result_accuracy, percentiles)):
        print(f"{percentiles[index]}: {percentile}", file=output_file)

    total_words = len(result_data)
    incorrect_part_of_speech = (
        total_words - correct_part_of_speech - unknown_part_of_speech
    )
    print(
        f"Total words: {total_words}, "
        f"total correct part of speech: {correct_part_of_speech}, "
        f"total incorrect part of speech: {incorrect_part_of_speech}, "
        f"error percent: {incorrect_part_of_speech / total_words * 100}, "
        f"correct percent: {correct_part_of_speech / total_words * 100}",
        file=output_file,
    )


def completeness(
    result_data: dataframe.Dataframe, output_file: io.TextIOWrapper
) -> None:
    print("\n========================\n", file=output_file)
    print("Completeness assessment", file=output_file)
    attribute_tree = attribute.load_attribute_tree()
    result_completeness = []
    total_non_zero = 0
    total_error = 0
    for row in result_data.read():
        fill, unfill = completeness_dfs(result_data.name_ids, "1", row, attribute_tree)
        non_zero = len([item for item in row if item != "0"])
        result_completeness.append(fill / (fill + unfill) * 100)

        total_non_zero += non_zero
        total_error = non_zero - fill

    percentiles = [5, 10, 25]
    print(f"Mean: {numpy.mean(result_completeness)}", file=output_file)
    print(f"Median: {numpy.median(result_completeness)}", file=output_file)
    for index, percentile in enumerate(
        numpy.percentile(result_completeness, percentiles)
    ):
        print(f"{percentiles[index]}: {percentile}", file=output_file)

    print("Error fillness", file=output_file)
    print(
        f"total non zero = {total_non_zero}, "
        f"total error = {total_error}, "
        f"error percentile = {total_error / total_non_zero * 100}",
        file=output_file,
    )


def completeness_dfs(
    names_ids: Mapping[str, int],
    root_id: str,
    row: List[str],
    attribute_tree: attribute.AttributeTree,
) -> Tuple[int, int]:
    root = attribute_tree[root_id]
    if root_id not in names_ids:
        return 0, 0
    attribute_value = row[names_ids[root_id]]
    if attribute_value == "0":
        return 0, root.vertices_to_nearest_leaf
    total_fill = total_unfill = 0
    for next_attribute in root.attribute_values[attribute_value].next_attributes:
        fill, unfill = completeness_dfs(names_ids, next_attribute, row, attribute_tree)
        total_fill += fill
        total_unfill += unfill
    return total_fill + 1, total_unfill


if __name__ == "__main__":
    compare()
