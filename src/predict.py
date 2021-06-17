import click
import os

from lib import (
    config as global_config,
    models,
    predict as global_predict,
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
@click.option("--output-directory", type=str, default="", help="Directory for output tsv")
@click.option("--output-filename", type=str, default="", help="")
def predict(config: str, meta: str, input: str, output_directory: str, output_filename: str) -> None:
    config = global_config.Config.from_file(config)
    model = models.create_model(config, meta)
    global_predict.predict(input, config=config, meta_path=meta).to_csv(
        os.path.join(
            output_directory or os.path.join("..", "models", model.model_id),
            output_filename or "predict.tsv"
        )
    )


if __name__ == "__main__":
    predict()
