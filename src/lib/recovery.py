import json
from typing import Any, List, Mapping

from lib import (
    attribute,
    dataframe,
    lemmers,
    predict,
)


def recovery_result(data: dataframe.Dataframe) -> str:
    interim = [
        {
            "word": row[data.name_ids["WORD"]],
            "word_id": row[data.name_ids["word_id"]],
            "sentence_id": row[data.name_ids["sentence_id"]],
            "paragraph_id": row[data.name_ids["paragraph_id"]],
            "initial_form": row[data.name_ids["initial_form"]],
            "result": recovery_dfs(
                attribute.load_attribute_tree(),
                "1",
                data.name_ids,
                row,
            ),
        }
        for row in data.read()
    ]
    return json.dumps(interim, indent=2, ensure_ascii=False)


def recovery_dfs(
    attribute_tree: attribute.AttributeTree,
    attribute_id: str,
    name_ids: Mapping[str, int],
    values: List[str],
) -> Mapping[str, Any]:
    if attribute_id not in name_ids:
        return {}
    attribute_value_id = values[name_ids[attribute_id]]
    if int(attribute_value_id) == 0:
        return {}
    attribute_value_name = (
        attribute_tree[attribute_id]
        .attribute_values[attribute_value_id]
        .attribute_value_name
    )
    result = {attribute_tree[attribute_id].attribute_name: attribute_value_name}
    children = {}
    for next_attribute_id in (
        attribute_tree[attribute_id]
        .attribute_values[attribute_value_id]
        .next_attributes
    ):
        children.update(
            recovery_dfs(attribute_tree, next_attribute_id, name_ids, values)
        )

    if children:
        result["Аттрибуты"] = children
    return result
