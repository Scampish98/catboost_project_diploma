import click
from typing import Optional

from lib.config import Config
from lib.models import create_model


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    help="Path to config in yaml",
)
@click.option("--meta", "-m", type=str, help="Path to meta in json")
@click.option(
    "--train-data",
    "-t",
    type=str,
    help="Path to train data in tsv",
)
@click.option(
    "--validation-data",
    "-v",
    type=str,
    default=None,
    help="Path to validation data in tsv",
)
def train(
    config: str, meta: str, train_data: str, validation_data: Optional[str]
) -> None:
    model = create_model(config=Config.from_file(config), meta_path=meta)

    if not model.smart_split:
        assert validation_data is not None
        model.train(
            meta_path=meta,
            train_data_file_path=train_data,
            validate_data_file_path=validation_data,
        )
    else:
        model.train(meta_path=meta, train_data_file_path=train_data)


if __name__ == "__main__":
    train()
