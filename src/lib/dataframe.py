from __future__ import annotations

import csv
import numpy as np
from copy import deepcopy
from functools import singledispatchmethod
from typing import (
    Any,
    AbstractSet,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

from catboost import Pool


from .logging import LoggedClass


class Dataframe(LoggedClass):
    def __init__(
        self,
        size: int,
        data: Iterable[Iterable[str]],
        names: Optional[Iterable[str]] = None,
        weights: Optional[Iterable[float]] = None,
        name_ids: Optional[MutableMapping[str, int]] = None,
        logging_level: str = "info",
    ) -> None:
        super(Dataframe, self).__init__(logging_level)
        self.size = size
        self.data = [list(line) for line in data]
        self.names = list(names) if names else None
        self.weights = list(weights) if weights else None
        self.name_ids = (
            {name: name_id for name_id, name in enumerate(self.names or [])}
            if name_ids is None
            else name_ids
        )
        self.logging_level = logging_level

    def read(self) -> Iterator[List[str]]:
        for row in zip(*self.data):
            yield list(row)

    def get_rows(self, row_ids: Union[List[int], AbstractSet[int]]) -> Dataframe:
        new_data = []
        for column in self.data:
            new_data.append([column[row_id] for row_id in row_ids])
        return Dataframe(
            len(row_ids),
            new_data,
            deepcopy(self.names),
            deepcopy(self.weights),
            deepcopy(self.name_ids),
            self.logging_level,
        )

    def add_row(self, row: List[str], names: List[str]) -> None:
        for name in names:
            if name not in self.name_ids:
                self.add_column(name, ["0"] * self.size)
        for name in self.names:
            if name not in names:
                self.data[self.name_ids[name]].append("0")
            else:
                self.data[self.name_ids[name]].append(row[names.index(name)])
        self.size += 1

    def add_rows(self, rows: List[List[str]], names: List[str]) -> None:
        for row in rows:
            self.add_row(row, names)

    def concatenate(self, data: Dataframe) -> None:
        for name in data.names:
            if name not in self.name_ids:
                self.add_column(name, ["0"] * self.size)
        for name in self.names:
            if name in data.name_ids:
                self.data[self.name_ids[name]] += data.data[data.name_ids[name]]
            else:
                self.data[self.name_ids[name]] += ["0"] * data.size
        self.size += data.size

    def filter_by_function(
        self,
        func: Callable[[List[str], Optional[List[str]], Optional[List[float]]], bool],
    ) -> Dataframe:
        result = [row for row in self.read() if func(row, self.names, self.weights)]
        return Dataframe(
            size=len(result),
            data=zip(*result),
            names=self.names,
            weights=self.weights,
            name_ids=self.name_ids,
            logging_level=self.logging_level,
        )

    def exclude_by_function(
        self,
        func: Callable[[List[str], Optional[List[str]], Optional[List[float]]], bool],
    ) -> Dataframe:
        result = [row for row in self.read() if not func(row, self.names, self.weights)]
        return Dataframe(
            size=len(result),
            data=zip(*result),
            names=self.names,
            weights=self.weights,
            name_ids=self.name_ids,
            logging_level=self.logging_level,
        )

    def split_by_function(
        self,
        func: Callable[[List[str], Optional[List[str]], Optional[List[float]]], bool],
    ) -> Tuple[Dataframe, Dataframe]:
        self._logger.info("Splitting dataframe started!")
        true_func = []
        false_func = []

        cnt = 0
        for row in self.read():
            if func(row, self.names, self.weights):
                true_func.append(row)
            else:
                false_func.append(row)
            cnt += 1
            if cnt % 1000 == 0:
                self._logger.info("%d complete", cnt)

        self._logger.info("Splitting dataframe finished!")

        return (
            Dataframe(
                size=len(true_func),
                data=zip(*true_func),
                names=self.names,
                weights=self.weights,
                name_ids=self.name_ids,
                logging_level=self.logging_level,
            ),
            Dataframe(
                size=len(false_func),
                data=zip(*false_func),
                names=self.names,
                weights=self.weights,
                name_ids=self.name_ids,
                logging_level=self.logging_level,
            ),
        )

    def get_columns(
        self,
        column_names: List[str],
        default_value: str = "0",
    ) -> Dataframe:
        new_data = []
        new_weights: Optional[List[float]] = [] if self.weights is not None else None
        for name in column_names:
            if name in self.name_ids:
                column_id = self.name_ids[name]
                new_data.append(deepcopy(self.data[column_id]))
                if new_weights is not None:
                    assert self.weights is not None
                    new_weights.append(self.weights[column_id])
            else:
                new_data.append([default_value] * self.size)
        return Dataframe(
            size=self.size,
            data=new_data,
            names=deepcopy(column_names),
            weights=new_weights,
            logging_level=self.logging_level,
        )

    def copy(self) -> Dataframe:
        return Dataframe(
            self.size,
            deepcopy(self.data),
            deepcopy(self.names),
            deepcopy(self.weights),
            deepcopy(self.name_ids),
            self.logging_level,
        )

    def get_catboost_pool(self, param_names: Iterable[str], target_name: str) -> Pool:
        temp_data = self[param_names]
        labels = self[target_name]
        self._logger.debug("temp_data: %s", temp_data)
        self._logger.debug("labels: %s", labels)
        return Pool(
            data=list(zip(*temp_data.data)),
            weight=temp_data.weights,
            label=labels,
            cat_features=list(range(len(temp_data.data))),
        )

    def get_sorted_by_name(self, key_name: str = "WORD") -> Dataframe:
        sorted_data = [row for row in self.read()]
        key_id = self.name_ids[key_name]
        sorted_data.sort(key=lambda row: row[key_id])
        return Dataframe(
            self.size,
            zip(*sorted_data),
            self.names,
            self.weights,
            self.name_ids,
            logging_level=self.logging_level,
        )

    def add_column(
        self,
        column_name: str,
        column_values: List[str],
        column_weight: Optional[float] = None,
    ) -> None:
        self.name_ids[column_name] = len(self.data)
        self.data.append(column_values)
        if self.names is not None:
            self.names.append(column_name)
        if column_weight is not None and self.weights is not None:
            self.weights.append(column_weight)

    def to_csv(self, output_file_path: str, delimiter: str = "\t") -> None:
        with open(output_file_path, "w", newline="", encoding="utf-8") as output_stream:
            writer = csv.writer(output_stream, delimiter=delimiter)
            if self.weights is not None:
                writer.writerow(self.weights)
            if self.names is not None:
                writer.writerow(self.names)
            for row in zip(*self.data):
                writer.writerow(row)

    def update(
        self,
        dict_data: Mapping[str, List[str]],
    ) -> None:
        for key, value in dict_data.items():
            self[key] = value

    def __len__(self) -> int:
        return self.size

    @singledispatchmethod
    def __getitem__(self, item: Any) -> Union[Dataframe, List[str]]:
        raise TypeError(f"Unexpected type item {type(item)}")

    @__getitem__.register
    def _getitem_by_id(self, item: str) -> List[str]:
        if item in self.name_ids:
            return self.data[self.name_ids[item]]
        else:
            raise IndexError(f"No such index '{item}'")

    @__getitem__.register
    def _getitem_by_index(self, item: int) -> List[str]:
        return self.data[item]

    @__getitem__.register(list)
    def _get_columns(self, item: List[str]) -> Dataframe:
        return self.get_columns(item)

    def __setitem__(self, key: str, value: List[str]) -> None:
        if key in self.name_ids:
            row_id = self.name_ids[key]
            self.data[row_id] = list(value)
        else:
            self.add_column(key, value)

    def __contains__(self, item: str) -> bool:
        return self.name_ids is not None and item in self.name_ids

    def __str__(self) -> str:
        temp_data: List[Union[List[float], List[str], List[List[str]]]] = []
        if self.weights is not None:
            temp_data.append(self.weights)
        if self.names is not None:
            temp_data.append(self.names)
        for row in zip(*self.data):
            temp_data.append(list(row))
        return str(np.array(temp_data))

    @classmethod
    def from_tsv(
        cls,
        file_path: str,
        delimiter: str = "\t",
        with_names: bool = True,
        with_weights: bool = False,
    ) -> Dataframe:
        with open(file_path, "r", encoding="utf8", newline="") as input_stream:
            tsv_reader = csv.reader(
                input_stream, delimiter=delimiter, lineterminator="\n"
            )
            weights = map(float, next(tsv_reader)) if with_weights else None
            names = next(tsv_reader) if with_names else None

            lines = list(zip(*tsv_reader))
            size = len(lines[0]) if len(lines) else 0
        return cls(size=size, data=lines, names=names, weights=weights)
