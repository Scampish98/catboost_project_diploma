from collections import defaultdict
from typing import List, Mapping, Optional

from .. import attribute, lemmers
from ..dataframe import Dataframe
from .base import Model


class SequentialLearningBasedOnAttributeTreeModel(Model):  # type: ignore
    def train(
        self,
        meta_path,
        train_data_file_path: str,
        validate_data_file_path: Optional[str] = None,
    ) -> None:
        input_data = self._get_dataset(train_data_file_path)
        input_data = self.clean_excluded(input_data)
        attribute_tree = attribute.load_attribute_tree()
        params = self.get_params()
        self._train_dfs(
            meta_path,
            input_data,
            list(range(len(input_data))),
            params,
            ["1"],
            attribute_tree,
        )

    def _train_dfs(
        self,
        meta_path: str,
        input_data: Dataframe,
        ids: List[int],
        params: List[str],
        prefix_target_names: List[str],
        attribute_tree: Mapping[str, attribute.Attribute],
    ) -> None:
        trained_model_name = "_".join(prefix_target_names)
        target_name = prefix_target_names[-1]

        if int(target_name) in self.config.excluded_attributes:
            self._logger.info("Skip %s from excluded_attributes", trained_model_name)
            return
        self._logger.info("Start train model %s", trained_model_name)
        trained_model = self.get_catboost_model(
            input_data.get_rows(ids),
            None,
            params,
            target_name,
            trained_model_name,
        )
        if trained_model:
            self.save(trained_model_name, trained_model)
        if trained_model_name not in self.completed_models:
            self.completed_models.append(trained_model_name)
            self.update_meta(meta_path)
        self._logger.info("Finish train model %s", trained_model_name)

        attribute_values = list(input_data[target_name])
        attribute_value_row_ids = defaultdict(list)

        for row_id in ids:
            attribute_value_row_ids[attribute_values[row_id]].append(row_id)

        for attribute_value_id, row_ids in attribute_value_row_ids.items():
            if int(attribute_value_id) != 0:
                for next_attribute_id in (
                    attribute_tree[target_name]
                    .attribute_values[str(attribute_value_id)]
                    .next_attributes
                ):
                    prefix_target_names.append(next_attribute_id)
                    self._train_dfs(
                        meta_path,
                        input_data,
                        row_ids,
                        params,
                        prefix_target_names,
                        attribute_tree,
                    )
                    prefix_target_names.pop()

    def predict_from_data(self, input_data: Dataframe) -> Dataframe:
        params = self.get_params()
        if not self.use_true_initial_form:
            input_data["initial_form"] = self.predict_initial_form(input_data["WORD"])
        self._logger.debug("input_data size: %s", input_data.size)
        russian, other, additional_params = self.split_by_heuristics(input_data)
        self._logger.debug("russian size: %s", russian.size)
        attribute_tree = attribute.load_attribute_tree()

        additional_data_rus = self.init_additional_data(russian, with_defaults=False)
        additional_data_oth = self.init_additional_data(other, with_defaults=False)

        russian = russian[params]
        other = other[params + additional_params]

        self._predict_dfs(
            russian,
            list(range(len(russian))),
            params,
            ["1"],
            attribute_tree,
        )

        other.update(additional_data_oth)
        russian.update(additional_data_rus)
        return russian + other

    def _predict_dfs(
        self,
        data: Dataframe,
        ids: List[int],
        params: List[str],
        prefix_target_names: List[str],
        attribute_tree: Mapping[str, attribute.Attribute],
    ) -> None:
        trained_model_name = "_".join(prefix_target_names)
        if trained_model_name not in self.completed_models:
            return
        target_name = prefix_target_names[-1]
        if target_name not in data:
            data[target_name] = ["0"] * len(data)

        self._logger.info("Start predict by %s", trained_model_name)

        if trained_model_name in self.default_values:
            model_result = [self.default_values[trained_model_name]] * len(ids)
        else:
            trained_model = self.load(trained_model_name)
            params_data = data[params].get_rows(ids)
            model_result = list(self.predict(params_data, trained_model))

        attribute_value_row_ids = defaultdict(list)
        for pos, row_id in enumerate(ids):
            data[target_name][row_id] = model_result[pos]
            attribute_value_row_ids[model_result[pos]].append(row_id)

        self._logger.info("Finish predict by %s", trained_model_name)

        for attribute_value_id, row_ids in attribute_value_row_ids.items():
            if int(attribute_value_id) != 0:
                for next_attribute_id in (
                    attribute_tree[target_name]
                    .attribute_values[str(attribute_value_id)]
                    .next_attributes
                ):
                    prefix_target_names.append(next_attribute_id)
                    self._predict_dfs(
                        data,
                        row_ids,
                        params,
                        prefix_target_names,
                        attribute_tree,
                    )
                    prefix_target_names.pop()
