from __future__ import print_function

import json
import os
from typing import cast, Any, List, Mapping, MutableMapping

import jsonpickle


ATTRIBUTE_TREE_PATH = os.path.join("..", "..", "data/attributes.json")


class AttributeValue:
    def __init__(
        self,
        attribute_value_id: str,
        attribute_value_name: str,
        next_attributes: List[str],
    ):
        self.attribute_value_id = attribute_value_id
        self.attribute_value_name = attribute_value_name
        self.next_attributes = next_attributes

    def as_json(self) -> MutableMapping[str, Any]:
        return self.__dict__.copy()

    def __str__(self) -> str:
        return json.dumps(self.as_json(), indent=2, ensure_ascii=False, sort_keys=True)


class Attribute:
    def __init__(
        self,
        attribute_id: str,
        attribute_name: str,
        attribute_values: Mapping[str, AttributeValue],
        vertices_to_nearest_leaf: int = 0,
    ):
        self.attribute_id = attribute_id
        self.attribute_name = attribute_name
        self.attribute_values = attribute_values
        self.vertices_to_nearest_leaf = vertices_to_nearest_leaf

    def as_json(self) -> MutableMapping[str, Any]:
        return self.__dict__.copy()

    def __str__(self) -> str:
        return cast(str, jsonpickle.dumps(self.as_json(), indent=2))


AttributeTree = Mapping[str, Attribute]


def load_attribute_tree() -> AttributeTree:
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ATTRIBUTE_TREE_PATH),
        "r",
        encoding="utf-8",
    ) as input_stream:
        return cast(Mapping[str, Attribute], jsonpickle.loads(input_stream.read()))
