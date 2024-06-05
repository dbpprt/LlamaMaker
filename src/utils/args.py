# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from ast import literal_eval
from typing import Dict, List


# source: https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/utils.py


def _convert_nargs_to_dict(nargs: List[str]) -> Dict[str, str]:
    if len(nargs) < 0:
        return {}
    # helper function to infer type for argsparser

    def _infer_type(s):
        try:
            s = float(s)

            if s // 1 == s:
                return int(s)
            return s
        except ValueError:
            return s

    parser = argparse.ArgumentParser()
    _, unknown = parser.parse_known_args(nargs)
    for index, argument in enumerate(unknown):
        if argument.startswith(("-", "--")):
            action = None
            if index + 1 < len(unknown):  # checks if next index would be in list
                if unknown[index + 1].startswith(("-", "--")):  # checks if next element is an key
                    # raise an error if element is store_true or store_false
                    raise ValueError(
                        "SageMaker doesn’t support argparse actions for `store_true` or `store_false`. Please define explicit types"
                    )
            else:  # raise an error if last element is store_true or store_false
                raise ValueError(
                    "SageMaker doesn’t support argparse actions for `store_true` or `store_false`. Please define explicit types"
                )
            # adds argument to parser based on action_store true
            if action is None:
                parser.add_argument(argument, type=_infer_type)
            else:
                parser.add_argument(argument, action=action)

    return {
        key: (literal_eval(value) if value in ("True", "False") else value)
        for key, value in parser.parse_args(nargs).__dict__.items()
    }


class _StoreAction(argparse.Action):
    """
    Custom action that allows for `-` or `_` to be passed in for an argument.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        new_option_strings = []
        for option_string in self.option_strings:
            new_option_strings.append(option_string)
            if "_" in option_string[2:]:
                # Add `-` version to the option string
                new_option_strings.append(option_string.replace("_", "-"))
        self.option_strings = new_option_strings

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class _StoreConstAction(_StoreAction):
    """
    Same as `argparse._StoreConstAction` but uses the custom `_StoreAction`.
    """

    def __init__(self, option_strings, dest, const, default=None, required=False, help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self.const)


class _StoreTrueAction(_StoreConstAction):
    """
    Same as `argparse._StoreTrueAction` but uses the custom `_StoreConstAction`.
    """

    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        required=False,
        help=None,
    ):
        super().__init__(
            option_strings=option_strings, dest=dest, const=True, default=default, required=required, help=help
        )


class CustomArgumentGroup(argparse._ArgumentGroup):
    """
    Custom argument group that allows for the use of `-` or `_` in arguments passed and overrides the help for each
    when applicable.
    """

    def _add_action(self, action):
        args = vars(action)
        if isinstance(action, argparse._StoreTrueAction):
            action = _StoreTrueAction(
                args["option_strings"], args["dest"], args["default"], args["required"], args["help"]
            )
        elif isinstance(action, argparse._StoreConstAction):
            action = _StoreConstAction(
                args["option_strings"],
                args["dest"],
                args["const"],
                args["default"],
                args["required"],
                args["help"],
            )
        elif isinstance(action, argparse._StoreAction):
            action = _StoreAction(**args)
        action = super()._add_action(action)
        return action


class CustomArgumentParser(argparse.ArgumentParser):
    """
    Custom argument parser that allows for the use of `-` or `_` in arguments passed and overrides the help for each
    when applicable.
    """

    def add_argument(self, *args, **kwargs):
        if "action" in kwargs:
            # Translate action -> class
            if kwargs["action"] == "store_true":
                kwargs["action"] = _StoreTrueAction
        else:
            kwargs["action"] = _StoreAction
        super().add_argument(*args, **kwargs)

    def add_argument_group(self, *args, **kwargs):
        group = CustomArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group
