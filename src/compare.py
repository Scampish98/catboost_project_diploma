import click

from lib import (
    dataframe,
)


@click.command()
@click.option("--result-data", type=str, help="Path to result data tsv")
@click.option("--validate-data", type=str, help="Path to validate data tsv")
def compare(result_data: str, validate_data: str) -> None:
    non_compared_names = {
        "WORD",
        "initial_form",
        "word_id",
        "sentence_id",
        "paragraph_id",
    }
    for word_id in range(1, 6):
        non_compared_names.add(f"word_{word_id}")
        non_compared_names.add(f"word_{-word_id}")
    result_data = dataframe.Dataframe.from_tsv(result_data).get_sorted_by_name("WORD")
    validate_data = dataframe.Dataframe.from_tsv(validate_data)[
        result_data.names
    ].get_sorted_by_name("WORD")
    total_cnt = 0
    total_bad_cnt = 0
    total_good_cnt = 0
    for name in result_data.names:
        if name in non_compared_names:
            continue
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
                f" good percent = {100.0 * current_good / current_total}"
            )
    print(
        f"Total bad = {total_bad_cnt},"
        f" total good = {total_good_cnt},"
        f" total cnt = {total_cnt},"
        f" total bad percent = {100.0 * total_bad_cnt / total_cnt}"
        f" total good percent = {100.0 * total_good_cnt / total_cnt}"
    )


if __name__ == "__main__":
    compare()
