from __future__ import annotations

import abc
import json
import os
import random
import uuid
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    MutableMapping,
    List,
    Optional,
    Tuple,
)

import jsonpickle
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from ..config import Config
from ..dataframe import Dataframe
from ..language_detector import LanguageDetector
from ..lemmers import get_initial_form
from ..logging import create_logger, LoggedClass, LOGGING_LEVELS


class Model(LoggedClass, abc.ABC):
    def __init__(
        self,
        model_id: str,
        config: Optional[Config] = None,
        completed_models: Optional[List[str]] = None,
        default_values: Optional[MutableMapping[str, str]] = None,
    ) -> None:
        self.model_id = model_id
        self.config = config or Config()
        self.completed_models = completed_models or []
        self.default_values = default_values or {}

        super().__init__(config.logging_level)

    @classmethod
    def from_config(cls, config: Config, meta_path: str) -> Model:
        key = config.as_key()
        meta = Meta(path=meta_path, logging_level=config.logging_level)
        if key in meta:
            model = meta[key]
            model.config = config
            super(Model, model).__init__(logging_level=config.logging_level)
        else:
            model = cls(
                model_id=meta[key].model_id if key in meta else str(uuid.uuid4()),
                config=config,
            )

        if config.refit_model:
            model.rewrite(meta_path)
            config.refit_model = False
            model.config.refit_model = False

        directory = model.directory_path
        os.makedirs(directory, exist_ok=True)

        model.config.to_file(os.path.join(directory, "config.yaml"))
        model.update_meta(meta_path)

        with create_logger() as logger:
            logger.info("model_id = %s", model.model_id)

        return model

    @abc.abstractmethod
    def train(
        self,
        meta_path: str,
        train_data_file_path: str,
        validate_data_file_path: Optional[str] = None,
    ) -> None:
        pass

    def predict_from_file(self, path: str) -> Dataframe:
        return self.predict_from_data(self._get_dataset(path))

    @abc.abstractmethod
    def predict_from_data(self, input_data: Dataframe) -> Dataframe:
        pass

    def predict_initial_form(self, words: List[str]) -> List[str]:
        return [get_initial_form(word, self.config.lemmer_type) for word in words]

    # @staticmethod
    def predict(self, params_data: Dataframe, trained_model: CatBoost) -> Iterator[str]:
        self._logger.debug("Predict params_data: %s", params_data)
        for item in trained_model.predict(params_data.get_catboost_pool()):
            yield str(int(item))

    def init_additional_data(
        self, data: Dataframe, with_defaults: bool = True
    ) -> MutableMapping[str, List[str]]:
        self._logger.debug("init_additional_data data type: %s", type(data))
        additional_data = {}
        total_len = len(data)
        if with_defaults:
            for name, value in self.default_values.items():
                additional_data[name] = [value] * total_len

        if not self.config.use_initial_form:
            additional_data["initial_form"] = list(data["initial_form"])

        for additional_data_name in [
            "text_id",
            "chapter_id",
            "word_id",
            "sentence_id",
            "paragraph_id",
            "true_part_of_speech",
        ]:
            try:
                additional_data[additional_data_name] = list(data[additional_data_name])
            except Exception as e:
                self._logger.error("Exception during parse additional data: %s", e)
                additional_data[additional_data_name] = ["-1"] * total_len
        return additional_data

    def rewrite(self, meta_path: str) -> None:
        self.completed_models = []
        self.default_values = {}
        self.update_meta(meta_path)

    def update_meta(self, meta_path: str) -> None:
        meta = Meta(path=meta_path, logging_level=self.logging_level)
        meta[self.key] = self
        meta.save()

    def get_catboost_model(
        self,
        train_data: Dataframe,
        validation_data: Optional[Dataframe],
        param_names: Iterable[str],
        target_name: str,
        target_name_for_default: str,
    ) -> CatBoost:
        if target_name_for_default in self.completed_models:
            trained_model = self.load(target_name_for_default)
        else:
            trained_model = None
        if not trained_model:
            trained_model = self.fit(
                train_data=train_data,
                validation_data=validation_data,
                param_names=param_names,
                target_name=target_name,
                target_name_for_default=target_name_for_default,
            )
        return trained_model

    def fit(
        self,
        train_data: Dataframe,
        validation_data: Optional[Dataframe],
        param_names: Iterable[str],
        target_name: str,
        target_name_for_default: str,
    ) -> Optional[CatBoost]:
        unique_values: List[str] = list(set(train_data[target_name]))
        if len(unique_values) == 1:
            self.default_values[target_name_for_default] = unique_values[0]
            return None

        self._logger.debug("Unique values: %s", unique_values)

        if validation_data is None:
            train, validation = self.split(train_data, target_name)
            train = train.get_catboost_pool(param_names, target_name)
            validation = validation.get_catboost_pool(param_names, target_name)
        else:
            train = train_data.get_catboost_pool(param_names, target_name)
            validation = validation_data.get_catboost_pool(param_names, target_name)

        trained_model = self.build(train_data[param_names])
        trained_model.fit(train, eval_set=validation)
        return trained_model

    def clean_excluded(self, data: Optional[Dataframe]) -> Optional[Dataframe]:
        if not data:
            return None
        self._logger.info("clean_excluded started!")
        result = [
            row
            for row in data.read()
            if not set(row).intersection(
                [
                    str(attribute_value)
                    for attribute_value in self.config.excluded_attribute_values
                ]
            )
        ]
        self._logger.info("clean_excluded finished!")
        return Dataframe(
            size=len(result),
            data=zip(*result),
            names=data.names,
            name_ids=data.name_ids,
            logging_level=self.logging_level,
        )

    def split_by_heuristics(
        self, data: Dataframe
    ) -> Tuple[Dataframe, Dataframe, List[str]]:
        (
            linguistic,
            non_linguistic,
            additional_linguistic_params,
        ) = self.split_by_non_linguistic(data)
        russian, other, additional_language_params = self.split_by_language(linguistic)
        return (
            russian,
            non_linguistic + other,
            additional_linguistic_params + additional_language_params,
        )

    def split_by_non_linguistic(
        self, data: Dataframe
    ) -> Tuple[Dataframe, Dataframe, List[str]]:
        linguistic = []
        non_linguistic = []
        for row_id, word in enumerate(data["WORD"]):
            if self._check_linguistic(word):
                linguistic.append(row_id)
            else:
                non_linguistic.append(row_id)
        non_linguistic = data.get_rows(non_linguistic)
        non_linguistic.update({"1": ["399"] * non_linguistic.size})
        return data.get_rows(linguistic), non_linguistic, ["1"]

    def _check_linguistic(self, word: str) -> bool:
        return word.replace("'", "").replace("-", "").replace("–", "").isalnum()

    def split_by_language(
        self, data: Dataframe
    ) -> Tuple[Dataframe, Dataframe, List[str]]:
        russian = []
        other = []
        other_result_data = defaultdict(list)
        detector = LanguageDetector(logging_level=self.logging_level)
        for row_id, iso in enumerate(detector.get_language_iso_batch(data["WORD"])):
            if iso == "ru":
                russian.append(row_id)
            else:
                other.append(row_id)
                other_result_data["1"].append("384")
                other_result_data["109"].append(detector.get_language_by_iso(iso)[1])
            if row_id % 1000 == 0:
                self._logger.info(f"{row_id} complete")
        other = data.get_rows(other)
        other.update(other_result_data)
        return data.get_rows(russian), other, ["1", "109"]

    @staticmethod
    def split(data: Dataframe, target_name: str) -> Tuple[Dataframe, Dataframe]:
        random.seed(target_name)

        train_ids = set()
        validate_ids = set()
        value_ids = defaultdict(list)

        target_values = list(data[target_name])
        for row_id in range(len(target_values)):
            train_ids.add(row_id)
            value_ids[target_values[row_id]].append(row_id)

        for value, ids in value_ids.items():
            cnt = len(ids)
            validate_count = int(cnt / 3) * 2
            for row_id in random.sample(ids, k=validate_count):
                train_ids.remove(row_id)
                validate_ids.add(row_id)

        if len(validate_ids) == 0:
            shifted_id = list(train_ids)[0]
            train_ids.remove(shifted_id)
            validate_ids.add(shifted_id)

        train_data = data.get_rows(train_ids)
        validate_data = data.get_rows(validate_ids)

        return train_data, validate_data

    def load(self, target_name: str) -> Optional[CatBoost]:
        trained_model = self.build()
        path = os.path.join(self.directory_path, self.get_name(target_name))
        if os.path.exists(path):
            trained_model.load_model(path)
            return trained_model
        return None

    def save(self, target_name: str, trained_model: CatBoost) -> None:
        os.makedirs(self.directory_path, exist_ok=True)
        trained_model.save_model(
            os.path.join(self.directory_path, self.get_name(target_name))
        )

    def build(self, data: Optional[Dataframe] = None) -> CatBoost:
        return CatBoostClassifier(**self.get_catboost_params(data))

    @staticmethod
    def get_name(name: str) -> str:
        return f"{name}.bin"

    @property
    def directory_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "models",
            self.model_id,
        )

    @property
    def key(self) -> str:
        return self.config.as_key()

    def as_json(self) -> MutableMapping[str, Any]:
        return self.__dict__.copy()

    def get_catboost_params(
        self, data: Optional[Dataframe] = None
    ) -> MutableMapping[str, Any]:
        catboost_params = self.config.catboost_params.copy()
        catboost_params["logging_level"] = LOGGING_LEVELS[self.logging_level].catboost
        return catboost_params

    def get_params(self) -> List[str]:
        params = ["WORD"]

        if self.config.use_initial_form:
            params.append("initial_form")
        for i in range(0, self.config.number_words_before):
            params.append(f"word_{-(i + 1)}")
        for i in range(0, self.config.number_words_after):
            params.append(f"word_{i + 1}")
        return params

    @property
    def logging_level(self) -> str:
        return self.config.logging_level

    @property
    def smart_split(self) -> bool:
        return self.config.smart_split

    @property
    def use_calculated_parameters(self) -> bool:
        return self.config.use_calculated_parameters

    @property
    def use_true_initial_form(self) -> Optional[float]:
        return self.config.use_true_initial_form

    def __str__(self) -> str:
        model_data = self.as_json()
        model_data["config"] = self.config.as_json()
        return json.dumps(model_data, sort_keys=True, ensure_ascii=False, indent=2)

    def _get_dataset(
        self,
        file_path: str,
        delimiter: str = "\t",
        with_names: bool = True,
    ) -> Dataframe:
        return Dataframe.from_tsv(
            file_path=file_path,
            delimiter=delimiter,
            with_names=with_names,
            logging_level=self.logging_level,
        )


class Meta(LoggedClass, Dict[str, Model]):
    def __init__(self, path: str, logging_level: str) -> None:
        super().__init__(logging_level)
        self.path = path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    meta_data = jsonpickle.loads(f.read())
                    for item in meta_data:
                        self[item.key] = item
            except Exception as e:
                self._logger.error("Exception during read meta from file: %s", e)

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(jsonpickle.dumps(list(self.values()), indent=2))
        except Exception as e:
            self._logger.error("Exception during write meta to file: %s", e)
