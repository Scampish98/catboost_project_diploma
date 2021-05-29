import abc
import os
from functools import partial
from typing import List, Optional

from ..dataframe import Dataframe
from ..filters import filter_language
from .base import Model


ALL_NAMES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "data",
    "all_names.tsv",
)


class BaseSequentialLearningModel(Model, abc.ABC):  # type: ignore
    def train(
        self,
        meta_path: str,
        train_data_file_path: str,
        validate_data_file_path: Optional[str] = None,
    ) -> None:
        self._logger.debug("train_data load started!")
        train_data = self._get_dataset(train_data_file_path)
        self._logger.debug("train_data load finished!")
        self._logger.debug("validation_data load started!")
        validation_data = (
            self._get_dataset(validate_data_file_path)
            if not self.smart_split or validate_data_file_path is not None
            else None
        )
        self._logger.debug("validation_data load finished!")
        train_data = self.clean_excluded(train_data)
        validation_data = self.clean_excluded(validation_data)
        self.set_calculated_parameters_weight(train_data)
        self.set_calculated_parameters_weight(validation_data)

        non_trained_names = self._get_non_trained_names()
        params = self.get_params()

        if self.config.refit_model:
            self.rewrite(meta_path)
        all_names = self._get_all_names()

        for name in all_names:
            if name not in self.completed_models and name not in non_trained_names:
                self._logger.info("Start train model %s", name)
                trained_model = self.get_catboost_model(
                    train_data,
                    validation_data,
                    params,
                    name,
                    name,
                )
                if trained_model:
                    self.save(name, trained_model)
                self.completed_models.append(name)
                self.update_meta(meta_path)
                self._logger.info("Finish train model %s", name)
            else:
                self._logger.info(
                    "%s in %s",
                    name,
                    "non_trained_names"
                    if name in non_trained_names
                    else "completed_models",
                )
            if (
                name not in non_trained_names
                and name not in self.default_values
                and self.use_calculated_parameters
            ):
                params.append(name)

    def _get_non_trained_names(self) -> List[str]:
        names = ["WORD", "initial_form", "word_id", "sentence_id", "paragraph_id"]
        for i in range(1, 6):
            names += [f"word_{i}", f"word_{-i}"]
        names += ["word_id", "sentence_id", "paragraph_id"]
        names += [str(attribute) for attribute in self.config.excluded_attributes]
        return names

    def predict_from_data(self, data: Dataframe) -> Dataframe:
        params = self.get_params()
        russian, other, additional_params = self.split_by_heuristics(data)

        additional_data_rus = self.init_additional_data(russian)
        other.update(self.init_additional_data(other))
        russian = russian[params]
        other = other[params + additional_params]

        for name in self.completed_models:
            self._logger.info("Start predict by model %s", name)
            if name not in self.default_values:
                trained_model = self.load(name)
                if trained_model:
                    add_data = [str(x) for x in self.predict(russian, trained_model)]
                    if not self.config.use_calculated_parameters:
                        additional_data_rus[name] = add_data
                    else:
                        russian[name] = add_data
                        params.append(name)
                else:
                    raise Exception(
                        f"{self.directory_path} or {self.get_name(name)} not exists"
                    )

            self._logger.info("Finish predict by model %s", name)

        russian.update(additional_data_rus)
        return russian + other

    @staticmethod
    @abc.abstractmethod
    def _get_all_names() -> List[str]:
        pass


class SequentialLearningModel(BaseSequentialLearningModel):
    @staticmethod
    def _get_all_names() -> List[str]:
        with open(
            ALL_NAMES_PATH,
            "r",
            encoding="utf-8",
        ) as input_stream:
            return list(input_stream.readline().strip().split("\t"))


class ReversedSequentialLearningModel(BaseSequentialLearningModel):
    @staticmethod
    def _get_all_names() -> List[str]:
        with open(
            ALL_NAMES_PATH,
            "r",
            encoding="utf-8",
        ) as input_stream:
            return list(input_stream.readline().strip().split("\t"))[::-1]
