import typing
from types import NoneType


# https://gist.github.com/jacksmith15/482c2109dc83f156ff2a3246b82f20a2
def infer_schema(schema: dict, document: typing.Union[dict, list, str, float, int, bool, None]) -> None:
    document_type = {
        str: "string",
        float: "number",
        int: "integer",
        bool: "boolean",
        NoneType: "null",
        dict: "object",
        list: "array",
    }[type(document)]

    any_of = schema.get("anyOf")
    if any_of:
        infer_schema(_extract_matching_any_of(any_of, document_type), document)
        return

    current_type = schema.get("type")
    if not current_type:
        schema["type"] = document_type
    elif current_type != document_type:
        sub_schema: dict = {}
        infer_schema({}, document)
        _replace_dict_inplace(schema, {"anyOf": [{**schema}, sub_schema]})
        return

    if isinstance(document, dict):
        for key, value in document.items():
            infer_schema(schema.setdefault("properties", {}).setdefault(key, {}), value)

    if isinstance(document, list):
        for item in document:
            infer_schema(schema.setdefault("items", {}), item)


def _extract_matching_any_of(any_of: typing.List[dict], document_type: str) -> dict:
    for sub_schema in any_of:
        if sub_schema.get("type") == document_type:
            return sub_schema
    sub_schema = {}
    any_of.append(sub_schema)
    return sub_schema


def _replace_dict_inplace(dictionary: dict, replacement: dict) -> None:
    for key in dictionary:
        del dictionary[key]
    for key, value in replacement.items():
        dictionary[key] = value
