import click

from lib import (
    attribute,
    config as global_config,
    dataframe,
    lemmers,
    predict as global_predict,
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
@click.option("--input", type=str, help="Path to input tsv")
@click.option("--output", type=str, help="Path to output tsv")
def predict(config: str, meta: str, input: str, output: str) -> None:
    global_predict.predict(
        input,
        config=global_config.Config.from_file(config),
        meta_path=meta,
    ).to_csv(output)


if __name__ == "__main__":
    predict()
