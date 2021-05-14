from functools import singledispatch
from typing import Any

from .config import Config
from .dataframe import Dataframe
from .models import create_model


@singledispatch
def predict(data: Any, config: Config, meta_path: str) -> Dataframe:
    raise TypeError(f"Unexpected data type {type(data)}")


@predict.register
def predict_from_file(data: str, config: Config, meta_path: str) -> Dataframe:
    return create_model(config=config, meta_path=meta_path).predict_from_file(data)


@predict.register
def predict_from_data(data: Dataframe, config: Config, meta_path: str) -> Dataframe:
    return create_model(config=config, meta_path=meta_path).predict_from_data(data)
