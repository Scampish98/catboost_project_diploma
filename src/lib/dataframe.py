from __future__ import annotations, print_function

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

from .logging import create_logger, LoggedClass


class Dataframe(LoggedClass):
    def __init__(
        self,
        size: int,
        data: Iterable[Iterable[str]],
        names: Optional[Iterable[str]] = None,
        name_ids: Optional[MutableMapping[str, int]] = None,
        logging_level: str = "info",
    ) -> None:
        super(Dataframe, self).__init__(logging_level)
        self.size = size
        self.data = [list(line) for line in data]
        self.names = list(names) if names else None
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
        return Dataframe(
            len(row_ids),
            [[column[row_id] for row_id in row_ids] for column in self.data],
            self.names.copy(),
            self.name_ids.copy(),
            self.logging_level,
        )

    def __add__(self, other: Dataframe) -> Dataframe:
        result = self.copy()
        result.concatenate(other)
        return result

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

    def get_columns(
        self,
        column_names: List[str],
        default_value: str = "0",
    ) -> Dataframe:
        self._logger.debug("names: %s", self.names)
        self._logger.debug("name_ids: %s", self.name_ids)
        self._logger.debug("column_names: %s", column_names)
        self._logger.debug("len(data): %s", len(self.data))
        column_names = list(sorted(set(column_names)))
        new_data = []
        for name in column_names:
            if name in self.name_ids:
                column_id = self.name_ids[name]
                self._logger.debug("column_id: %s", column_id)
                new_data.append(self.data[column_id].copy())
            else:
                new_data.append([default_value] * self.size)
        return Dataframe(
            size=self.size,
            data=new_data,
            names=column_names.copy(),
            logging_level=self.logging_level,
        )

    def copy(self) -> Dataframe:
        return Dataframe(
            self.size,
            deepcopy(self.data),
            self.names.copy(),
            self.name_ids.copy(),
            self.logging_level,
        )

    def get_catboost_pool(
        self,
        param_names: Optional[Iterable[str]] = None,
        target_name: Optional[str] = None,
    ) -> Pool:
        temp_data = self[param_names] if param_names is not None else self.copy()
        labels = self[target_name] if target_name is not None else None
        self._logger.debug("temp_data: %s", temp_data)
        self._logger.debug("labels: %s", labels)

        return Pool(
            data=list(zip(*temp_data.data)),  # .get_sorted_data_by_column_names())),
            label=labels,
            feature_names=temp_data.names,
            cat_features=temp_data.names,
        )

    def get_sorted_data_by_column_names(self) -> List[List[str]]:
        return [self.data[self.name_ids[name]] for name in sorted(self.names)]

    def sort(self, columns: List[str]) -> None:
        sorted_data = [row for row in self.read()]
        keys = [self.name_ids[name] for name in columns]
        sorted_data.sort(key=lambda row: [int(row[key]) for key in keys])
        self.data = [list(item) for item in zip(*sorted_data)]

    def get_sorted_by_column_names(self) -> Dataframe:
        names = sorted(self.names)
        return Dataframe(
            size=self.size,
            data=[self.data[self.name_ids[name]] for name in names],
            names=names,
        )

    def get_sorted_by_name(self, key_name: str = "WORD") -> Dataframe:
        sorted_data = [row for row in self.read()]
        key_id = self.name_ids[key_name]
        sorted_data.sort(key=lambda row: row[key_id])
        return Dataframe(
            self.size,
            zip(*sorted_data),
            self.names,
            self.name_ids,
            logging_level=self.logging_level,
        )

    def add_column(
        self,
        column_name: str,
        column_values: List[str],
    ) -> None:
        self.name_ids[column_name] = len(self.data)
        self.data.append(column_values)
        if self.names is not None:
            self.names.append(column_name)

    def to_csv(self, output_file_path: str, delimiter: str = "\t") -> None:
        with open(output_file_path, "w", newline="", encoding="utf-8") as output_stream:
            writer = csv.writer(output_stream, delimiter=delimiter)
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

    def __setitem__(
        self, key: str, value: List[str]
    ) -> None:
        if key in self.name_ids:
            row_id = self.name_ids[key]
            self.data[row_id] = list(value)
        else:
            self.add_column(key, value)

    def __contains__(self, item: str) -> bool:
        return self.name_ids is not None and item in self.name_ids

    def __str__(self) -> str:
        temp_data: List[Union[List[float], List[str], List[List[str]]]] = []
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
        logging_level: str = "info",
    ) -> Dataframe:
        with open(file_path, "r", encoding="utf8", newline="") as input_stream:
            tsv_reader = csv.reader(
                input_stream, delimiter=delimiter, lineterminator="\n"
            )
            names = list(next(tsv_reader)) if with_names else None

            with create_logger("info") as logger:
                logger.info("names: %s", names)
            lines = list(zip(*tsv_reader))
            size = len(lines[0]) if len(lines) else 0
        return cls(
            size=size,
            data=lines,
            names=names,
            logging_level=logging_level,
        )
