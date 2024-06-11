import argparse
import os
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path

import sagemaker
from sagemaker import s3_utils
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.pytorch import PyTorch

from src.constants import (
    SAGEMAKER_CODE_TAR_GZ,
    SAGEMAKER_SUPPORTED_INSTANCES_AND_GPUS,
    SM_TENSORBOARD_OUTPUT_DIRECTORY,
)
from src.utils.args import CustomArgumentParser, _convert_nargs_to_dict
from src.utils.misc import run_command, s3_combine_url, s3_url_ensure_trailing_slash


def package_code(script_dir: str, s3_base: str) -> str:
    if script_dir:
        if not os.path.isabs(script_dir):
            script_dir = os.path.join(os.getcwd(), script_dir)
    else:
        script_dir = ""

    with tempfile.TemporaryDirectory() as tmp_path:
        tmp_tar_path = os.path.join(tmp_path, SAGEMAKER_CODE_TAR_GZ)

        print(f"Packaging script_dir {script_dir} to {tmp_tar_path} (ignoring files ignored by git)")

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
                    print(f"Adding {path} to {tmp_tar_path}")
                    tar.add(path)

        print(f"Synchronizing code to target directory {s3_base}")
        run_command(
            ["aws", "s3", "sync", tmp_path, s3_base],
            shell=False,
            env=os.environ,
            cwd=None,
        )

    return s3_combine_url(s3_base, SAGEMAKER_CODE_TAR_GZ)


def launch_command_parser(subparsers=None):
    description = "Launch a python script in a distributed scenario. Arguments can be passed in with either hyphens (`--num-processes=2`) or underscores (`--num_processes=2`)"
    if subparsers is not None:
        parser = subparsers.add_parser("launch", description=description, add_help=False, allow_abbrev=False)
    else:
        parser = CustomArgumentParser(
            "LlamaMaker launch command",
            description=description,
            add_help=False,
            allow_abbrev=False,
        )

    parser.add_argument(
        "--remote_config_file",
        default=None,
        help="The config file to use for the default values in the launching script on the remote machine.",
    )

    parser.add_argument("--base_job_name", default="llamamaker-", help="The base job name to use for the launching")
    parser.add_argument(
        "--s3_bucket",
        default=None,
        help="The S3 bucket to use as the base for the code and tensorboard outputs. Uses SageMaker default bucket if not provided.",
    )
    parser.add_argument(
        "--s3_bucket_prefix",
        default="",
        help="The S3 bucket prefix to use as the base for the code and tensorboard outputs.",
    )
    parser.add_argument(
        "--ec2_instance_type", default=None, help="The EC2 instance type to use for the launching", required=True
    )
    parser.add_argument(
        "--iam_role_name", default=None, help="The IAM role name to use for the launching", required=True
    )
    parser.add_argument(
        "--profile", default=None, help="The AWS profile name to use for the launching", required=False
    )
    parser.add_argument("--aws_access_key_id", default=None, help="The AWS access key ID to use for the launching")
    parser.add_argument(
        "--aws_secret_access_key", default=None, help="The AWS secret access key to use for the launching"
    )
    parser.add_argument("--num_machines", default=1, type=int, help="The number of machines to use for the launching")
    parser.add_argument("--region", default=None, help="The AWS region to use for the launching", required=True)
    parser.add_argument(
        "--image_uri", default=None, help="The image URI (ECR) to use as training container", required=True
    )
    parser.add_argument(
        "--sagemaker_inputs_file",
        default=None,
        help="The SageMaker inputs file to extract input data from the training job",
    )
    parser.add_argument(
        "--sagemaker_metrics_file",
        default=None,
        help="The SageMaker metrics file to extract metrics from the training job",
    )

    parser.add_argument(
        "training_script",
        type=str,
        help=(
            "The full path to the script to be launched in parallel, followed by all the arguments for the training "
            "script."
        ),
    )

    # Other arguments of the training scripts
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER, help="Arguments of the training script.")

    if subparsers is not None:
        parser.set_defaults(func=launch_command)
    return parser


def launch_command(args):
    # configure environment
    print("Configuring Amazon SageMaker environment")
    os.environ["AWS_DEFAULT_REGION"] = args.region

    instance_type = args.ec2_instance_type
    assert (
        instance_type in SAGEMAKER_SUPPORTED_INSTANCES_AND_GPUS
    ), f"The instance type {instance_type} is not supported."
    num_gpus = SAGEMAKER_SUPPORTED_INSTANCES_AND_GPUS[instance_type]
    print(f"The instance type {instance_type} has {num_gpus} GPU(s).")

    # configure credentials
    if args.profile is not None:
        os.environ["AWS_PROFILE"] = args.profile
    elif args.aws_access_key_id is not None and args.aws_secret_access_key is not None:
        os.environ["AWS_ACCESS_KEY_ID"] = args.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = args.aws_secret_access_key
    else:
        raise OSError("You need to provide an aws_access_key_id and aws_secret_access_key when not using profile")

    session = sagemaker.Session()
    bucket, _ = s3_utils.determine_bucket_and_prefix(bucket=None, key_prefix=None, sagemaker_session=session)

    if args.s3_bucket is not None:
        bucket = s3_url_ensure_trailing_slash(args.s3_bucket)
        print(f"Using bucket {args.s3_bucket} as the S3 bucket")
    else:
        bucket = f"s3://{bucket}/"
        print(f"S3 bucket not provided, using bucket {bucket} as the S3 bucket")

    s3_base_path = s3_url_ensure_trailing_slash(
        s3_combine_url(bucket, s3_url_ensure_trailing_slash(args.s3_bucket_prefix))
    )

    print(f"Base directory for this launch is {s3_base_path}")

    # TODO: Using just a uuid here is unfortunate, it should be the SageMaker job name, which we have to preliminary calculate
    s3_script_path = package_code(
        ".",  # let's for now use the current working directory as the base path
        s3_base=s3_combine_url(s3_base_path, s3_url_ensure_trailing_slash(str(uuid.uuid4()))),
    )

    entry_point = os.path.basename(args.training_script)
    if not entry_point.endswith(".py"):
        raise ValueError(f'Your training script should be a python script and not "{entry_point}"')

    print("Converting Arguments to Hyperparameters")
    hyperparameters = _convert_nargs_to_dict(args.training_script_args)

    # configure sagemaker inputs
    sagemaker_inputs = None
    if args.sagemaker_inputs_file is not None:
        print(f"Loading SageMaker Inputs from {args.sagemaker_inputs_file} file")
        sagemaker_inputs = {}
        with open(args.sagemaker_inputs_file) as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                l = line.split("\t")
                sagemaker_inputs[l[0]] = l[1].strip()
        print(f"Loaded SageMaker Inputs: {sagemaker_inputs}")

    # configure sagemaker metrics
    sagemaker_metrics = None
    if args.sagemaker_metrics_file is not None:
        print(f"Loading SageMaker Metrics from {args.sagemaker_metrics_file} file")
        sagemaker_metrics = []
        with open(args.sagemaker_metrics_file) as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                l = line.split("\t")
                metric_dict = {
                    "Name": l[0],
                    "Regex": l[1].strip(),
                }
                sagemaker_metrics.append(metric_dict)
        print(f"Loaded SageMaker Metrics: {sagemaker_metrics}")

    # we need to pass the remote_config_file as config_file for the remote accelerate script to work
    hyperparameters["config_file"] = args.remote_config_file

    # configure session
    print("Creating Estimator")
    args = {
        "image_uri": args.image_uri,
        # we use our custom launcher here as "middleware" to call accelerate correctly
        "entry_point": "remote_launcher.py",
        "source_dir": s3_script_path,
        "role": args.iam_role_name,
        "base_job_name": args.base_job_name,
        "instance_count": args.num_machines,
        "instance_type": args.ec2_instance_type,
        "debugger_hook_config": False,
        "hyperparameters": hyperparameters,
        "environment": {"ENTRYPOINT": entry_point, "NUM_GPUS": num_gpus},
        "metric_definitions": sagemaker_metrics,
        "enable_sagemaker_metrics": True,
        "tensorboard_output_config": TensorBoardOutputConfig(
            s3_output_path=s3_combine_url(s3_base_path, "tensorboard"),
            container_local_output_path=SM_TENSORBOARD_OUTPUT_DIRECTORY,
        ),
    }

    # TODO: allow this again, currently not possible.
    # if sagemaker_config.additional_args is not None:
    #     args = merge_dicts(sagemaker_config.additional_args, args)

    huggingface_estimator = PyTorch(**args)
    huggingface_estimator.fit(inputs=sagemaker_inputs)
    print(f"You can find your model data at: {huggingface_estimator.model_data}")


def main():
    parser = CustomArgumentParser(
        "LlamaMaker SageMaker launcher", usage="launcher <command> [<args>]", allow_abbrev=False
    )
    subparsers = parser.add_subparsers(help="LlamaMaker command helpers")

    # register commands
    launch_command_parser(subparsers=subparsers)

    # let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # run
    args.func(args)


if __name__ == "__main__":
    main()
