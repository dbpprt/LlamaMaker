import json
import re
import traceback
from typing import Optional


def formatting_func(prompt: str, eos_token: Optional[str] = None, json_fields: Optional[list] = None):
    # we get all variables in curly brackets but skip double curly brackets
    variables_in_prompt = re.findall(r"(?<!{)({(?!{).*?}(?!}))", prompt)
    # remove curly brackets if any (regex is not fully working)
    variables_in_prompt = [f.strip("{}") for f in variables_in_prompt]

    def _formatting_func(examples):
        try:
            texts = []
            for i in range(len(examples[variables_in_prompt[0]])):
                f_dict = {k: examples[k][i] for k in variables_in_prompt}

                # verify json integrity
                for k in variables_in_prompt:
                    if json_fields and k in json_fields:
                        try:
                            _ = json.loads(f_dict[k])
                        except:  # noqa: E722
                            raise ValueError(f"Invalid json in {k}, with value {f_dict[k]}")

                text = prompt.format(**f_dict)

                if eos_token:
                    text += eos_token
                texts.append(text)
            return texts
        except:  # noqa: E722
            traceback.print_exc()

    return _formatting_func
