from typing import Mapping, Type

from ..config import Config
from .base import Model
from .sequential_learning_based_on_attribute_tree_model import (
    SequentialLearningBasedOnAttributeTreeModel,
)
from .sequential_learning_model import (
    SequentialLearningModel,
    ReversedSequentialLearningModel,
)


model_versions: Mapping[int, Type[Model]] = {
    0: SequentialLearningModel,
    1: SequentialLearningBasedOnAttributeTreeModel,
    2: ReversedSequentialLearningModel,
}


def create_model(config: Config, meta_path: str) -> Model:
    return model_versions[config.model_version].from_config(
        config=config, meta_path=meta_path
    )
