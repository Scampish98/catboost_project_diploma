from __future__ import annotations

import json
import yaml
from typing import Any, List, MutableMapping, Optional

from .logging import create_logger, LoggedClass


class Config(LoggedClass):
    def __init__(
        self,
        iterations: int = 100,
        loss_function: Optional[str] = None,
        learning_rate: Optional[float] = None,
        model_type: str = "classifier",
        task_type: str = "CPU",
        number_words_before: int = 5,
        number_words_after: int = 5,
        use_initial_form: bool = True,
        smart_split: bool = True,
        model_version: int = 0,
        use_calculated_parameters: bool = True,
        refit_model: bool = False,
        logging_level: str = "stats",
        lemmer_type: str = "smalt_stemmer",
        language_filter: bool = True,
        excluded_attributes: Optional[List[int]] = None,
        excluded_attribute_values: Optional[List[int]] = None,
    ):
        super().__init__(logging_level)
        self.catboost_params = {
            "iterations": iterations,
            "loss_function": loss_function,
            "learning_rate": learning_rate,
            "task_type": task_type,
        }
        self.model_type = model_type
        self.number_words_before = max(min(number_words_before, 5), 0)
        self.number_words_after = max(min(number_words_after, 5), 0)
        self.use_initial_form = use_initial_form
        self.smart_split = smart_split
        self.model_version = model_version
        self.use_calculated_parameters = use_calculated_parameters
        self.refit_model = refit_model
        self.logging_level = logging_level
        self.lemmer_type = lemmer_type
        self.language_filter = language_filter
        self.excluded_attributes = excluded_attributes or []
        self.excluded_attribute_values = excluded_attribute_values or []

    @classmethod
    def from_file(cls, path: str) -> Config:
        with open(path, "r", encoding="utf-8") as input_stream:
            try:
                config_data = yaml.load(input_stream, Loader=yaml.SafeLoader)
                config_data.update(config_data.pop("catboost_params"))
                return cls(**config_data)
            except Exception as e:
                with create_logger() as logger:
                    logger.error(
                        "Exception during initialization config from file: %s", e
                    )
                raise

    def as_key(self) -> str:
        key_data = self.as_json()
        key_data.pop("refit_model", None)
        key_data.pop("logging_level", None)
        key_data.pop("_logger", None)
        return json.dumps(key_data, sort_keys=True)

    def as_json(self) -> MutableMapping[str, Any]:
        return self.__dict__.copy()

    def __str__(self) -> str:
        return json.dumps(self.as_json(), sort_keys=True, ensure_ascii=False, indent=2)

    def to_file(self, path: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as output_stream:
                yaml.dump(self.as_json(), output_stream)
        except Exception as e:
            self._logger.error("Exception during save config to file: %s", e)
