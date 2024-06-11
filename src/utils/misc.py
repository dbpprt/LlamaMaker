import subprocess
from functools import reduce
from typing import Any, List, Tuple
from urllib.parse import urljoin

import torch


def merge_dicts(source, destination):
    """
    Recursively merges two dictionaries.

    Args:
        source (`dict`): The dictionary to merge into `destination`.
        destination (`dict`): The dictionary to merge `source` into.
    """
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            merge_dicts(value, node)
        else:
            destination[key] = value

    return destination


def is_ampere_or_newer(device_id=0):
    """Check if a GPU supports FlashAttention."""
    major, minor = torch.cuda.get_device_capability(device_id)

    # Check if the GPU architecture is Ampere (SM 8.x) or newer (SM 9.0)
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

    return is_sm8x or is_sm90


def s3_combine_url(*args: List[str]) -> str:
    parts = list(args)
    s3_prefix = "s3://"
    prefix_stripped = False
    if parts[0].startswith(s3_prefix):
        parts[0] = parts[0][len(s3_prefix) :]
        prefix_stripped = True
    result = reduce(lambda a, b: urljoin(a, b, allow_fragments=False), parts)
    if prefix_stripped:
        return f"{s3_prefix}{result}"
    return result


def s3_url_ensure_trailing_slash(url: str) -> str:
    if url.endswith("/"):
        return url
    return f"{url}/"


def run_command(args: Any, shell: bool, env: Any, cwd: Any) -> Tuple[str, str]:
    with subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        env=env,
        cwd=cwd,
    ) as proc:
        print(f"Running command: {' '.join(args)}")
        out, err = proc.communicate()
        out_str = out.decode().strip() if out is not None else ""
        err_str = err.decode().strip() if err is not None else ""
        print(f"Output: {out_str} \n Error: {err_str if err_str != '' else 'No errors occurred!'}")
        return out_str, err_str
