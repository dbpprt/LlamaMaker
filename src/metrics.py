import json
import re
from collections.abc import MutableMapping

import numpy as np
from jsonschema import validate

from src.utils import infer_schema


# 1/ json parseable
# 2/ json has correct schema


# let's don't do any sampling here and just stick to argmax
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def flatten(dictionary, parent_key=False, separator="."):
    """
    Turn a nested dictionary into a flattened dictionary
    src: https://stackoverflow.com/questions/51359783/how-to-flatten-multilevel-nested-json
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            if not value.items():
                items.append((new_key, None))
            else:
                items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, list):
            if len(value):
                for k, v in enumerate(value):
                    items.extend(flatten({str(k): v}, new_key, separator).items())
            else:
                items.append((new_key, None))
        else:
            items.append((new_key, value))
    return dict(items)


def parse_preds(preds, tokenizer):
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # decode generated summaries into text
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    json_preds = []
    for pred in preds:
        match = re.search(r"```json(.*?)```", pred, re.DOTALL)
        if match:
            json_preds.append(match.group(1))
        else:
            json_preds.append(pred)

    # split decoded preds on ``` only take the first part
    json_preds = [re.split(r"```", json_pred)[0] for json_pred in json_preds if json_preds != ""]
    # trim decoded preds
    json_preds = [json_pred.strip() for json_pred in json_preds if json_pred != ""]

    return json_preds


def compute_metrics(model, tokenizer):
    def _compute_metrics(eval_preds):
        preds, labels = eval_preds

        preds = parse_preds(preds, tokenizer)
        labels = parse_preds(labels, tokenizer)

        # we should also cache this, it should be static
        labels = [json.loads(label) for label in labels]

        state = {
            "valid_json": 0,
            "valid_schema": 0,
            "valid_samples": [],
            "invalid_samples": [],
        }

        field_accuracy = {}

        for idx, pred in enumerate(preds):
            try:
                pred = json.loads(pred)
                state["valid_json"] += 1

                label = labels[idx]

                desired_schema = {}
                # we should probably cache this, super slow.
                infer_schema(desired_schema, pred)

                try:
                    validate(pred, schema=desired_schema)
                    state["valid_schema"] += 1
                except:  # noqa: E722
                    raise ValueError(f"Schema validation failed for {pred}")

                pred = flatten(pred)
                label = flatten(label)

                for key in label.keys():
                    state_key = f"accuracy_{key}"

                    if state_key not in field_accuracy:
                        field_accuracy[state_key] = {
                            "correct": 0,
                            "total": 0,
                        }

                    field_accuracy[state_key]["total"] += 1

                    if key in pred and pred[key] == label[key]:
                        field_accuracy[state_key]["correct"] += 1

                state["valid_samples"].append(pred)
            except:  # noqa: E722
                state["invalid_samples"].append(pred)

        for key, value in field_accuracy.items():
            field_accuracy[key] = value["correct"] / value["total"]

        return {
            **field_accuracy,
            "valid_json": state["valid_json"] / len(labels),
            "valid_schema": state["valid_schema"] / len(labels),
        }

    return _compute_metrics

    # if isinstance(preds, tuple):
    #     preds = preds[0]

    # # Replace -100 in the preds as we can't decode them
    # preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # # Decode generated summaries into text
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # # Replace -100 in the labels as we can't decode them
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # parse the json between markdown ```json ... ```
    # decoded_preds_json = []
    # for decoded_pred in decoded_preds:
    #     match = re.search(r"```json(.*?)```", decoded_pred, re.DOTALL)
    #     if match:
    #         decoded_preds_json.append(match.group(1))
    #     else:
    #         decoded_preds_json.append(decoded_pred)

    # # split decoded preds on ``` only take the first part
    # decoded_preds_json = [
    #     re.split(r"```", decoded_pred)[0] for decoded_pred in decoded_preds_json if decoded_pred != ""
    # ]
    # # trim decoded preds
    # decoded_preds_json = [decoded_pred.strip() for decoded_pred in decoded_preds_json if decoded_pred != ""]
