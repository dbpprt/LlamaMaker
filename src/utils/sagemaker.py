import logging
import os
import subprocess
import sys
import tarfile
import tempfile
from functools import reduce
from pathlib import Path
from typing import Any, List, Tuple
from urllib.parse import urljoin

import psutil
from sagemaker.estimator import _TrainingJob


log = logging.getLogger(__name__)

CODE_TAR_GZ = "code.tar.gz"


def _package_code(script_dir: str, s3_base: str) -> str:
    script_dir = _get_abs_code_dir(script_dir)

    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_tar_path = os.path.join(tmp_path, CODE_TAR_GZ)

        log.info(f"Packaging script_dir {script_dir} to {tmp_tar_path} (ignoring files ignored by git)")

        # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
        git_dir_path = Path(
            subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
        ).resolve()

        with tarfile.open(tmp_tar_path, "w:gz") as tar:
            for path in Path(script_dir).resolve().rglob("*"):
                if (
                    path.is_file()
                    # ignore files in .git
                    and not str(path).startswith(str(git_dir_path))  # noqa: W503
                    # ignore files ignored by git
                    and (  # noqa: W503
                        subprocess.run(["git", "check-ignore", "-q", str(path)]).returncode == 1
                    )
                ):
                    path = os.path.relpath(path)
                    log.info(f"Adding {path} to {tmp_tar_path}")
                    tar.add(path)

            # including the hydra_plugin itself
            # there is likely a better way for that :/
            # remote_invoke_script_path = os.path.join(
            #     os.path.dirname(__file__), REMOTE_INVOKE_SCRIPT_NAME
            # )

            # tar.add(remote_invoke_script_path, arcname=REMOTE_INVOKE_SCRIPT_NAME)

        log.info(f"Synchronizing code to target directory {s3_base}")
        _run_command(
            ["aws", "s3", "sync", tmp_path, s3_base],
            shell=False,
            env=os.environ,
            cwd=None,
        )

    return _s3_combine_url(s3_base, CODE_TAR_GZ)


# def _run_sync(definitions: List[S3SyncConf], s3_base: str) -> None:
#     # sanity check
#     assert not any(
#         [definition.target_dir is None and definition.s3_prefix is None for definition in definitions]
#     ), "Either target_dir or s3_prefix has to be set!"

#     for definition in definitions:
#         log.info(f"Executing step: {definition.name}")

#         source_dir = _get_abs_code_dir(definition.source_dir)
#         if definition.target_dir is not None:
#             target_dir = definition.target_dir
#         else:
#             target_dir = _s3_url_ensure_trailing_slash(_s3_combine_url(s3_base, definition.s3_prefix))

#         log.info(f"Synchronizing {source_dir} to target sweep directory {target_dir}")

#         command = ["aws", "s3", "sync", source_dir, target_dir]

#         if definition.include is not None and any(definition.include):
#             command.extend(
#                 reduce(
#                     lambda a, b: a + b,
#                     [["--include", f"{x}"] for x in definition.include],
#                 )
#             )

#         if definition.exclude is not None and any(definition.exclude):
#             command.extend(
#                 reduce(
#                     lambda a, b: a + b,
#                     [["--exclude", f"{x}"] for x in definition.exclude],
#                 )
#             )

#         _run_command(command, shell=False, env=os.environ, cwd=source_dir)


# def _run_commands(commands: List[CommandConf]) -> None:
#     cwd = _get_abs_code_dir(".")
#     for command in commands:
#         log.info(f"Executing step: {command.name}")

#         env = os.environ.copy()
#         if command.extend_path is not None and any(command.extend_path):
#             env["PATH"] += os.pathsep + os.pathsep.join(command.extend_path)

#         _run_command(args=command.cmd, shell=command.shell, env=env, cwd=cwd)


def restart_program():
    """Restarts the current program, with file objects and descriptors
    cleanup
    """

    try:
        p = psutil.Process(os.getpid())
        for handler in p.get_open_files() + p.connections():
            os.close(handler.fd)
    except Exception as e:
        logging.error(e)

    python = sys.executable
    os.execl(python, python, *sys.argv)


def _run_command(args: Any, shell: bool, env: Any, cwd: Any) -> Tuple[str, str]:
    with subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=shell,
        env=env,
        cwd=cwd,
    ) as proc:
        log.info(f"Running command: {' '.join(args)}")
        out, err = proc.communicate()
        out_str = out.decode().strip() if out is not None else ""
        err_str = err.decode().strip() if err is not None else ""
        log.info(f"Output: {out_str} \n Error: {err_str if err_str != '' else 'No errors occurred!'}")
        return out_str, err_str


def _get_abs_code_dir(code_dir: str) -> str:
    if code_dir:
        if os.path.isabs(code_dir):
            return code_dir
        else:
            return os.path.join(os.getcwd(), code_dir)
    else:
        return ""


def _s3_combine_url(*args: List[str]) -> str:
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


def _s3_url_ensure_trailing_slash(url: str) -> str:
    if url.endswith("/"):
        return url
    return f"{url}/"


def _describe_job(job: _TrainingJob):
    return job.sagemaker_session.sagemaker_client.describe_training_job(TrainingJobName=job.job_name)


# def _parse_job_status(job: Job) -> JobStatus:
#     if job._last_training_job_description is not None:
#         sm_status = job._last_training_job_description["TrainingJobStatus"]

#         if sm_status in ["InProgress", "Stopping"]:
#             return JobStatus.UNKNOWN
#         elif sm_status in ["Completed"]:
#             return JobStatus.COMPLETED
#         else:
#             return JobStatus.FAILED
#     elif job.result is not None:
#         return job.result.status
#     else:
#         return JobStatus.UNKNOWN


# def _job_status_to_str(status: JobStatus) -> str:
#     if status == JobStatus.UNKNOWN:
#         return "Running/Unknown"
#     elif status == JobStatus.FAILED:
#         return "Failed"
#     else:
#         return "Completed"


def _split_s3_path(path: str) -> Tuple[str, str]:
    path_parts = path.replace("s3://", "").split("/")
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)
    return bucket, key
