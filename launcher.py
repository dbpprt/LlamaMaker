import argparse
import json
import os

from sagemaker.huggingface import HuggingFace

from accelerate.commands.config.config_args import SageMakerConfig
from src.utils.args import CustomArgumentParser, _convert_nargs_to_dict
from src.utils.misc import merge_dicts


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
        "--config_file",
        default=None,
        help="The config file to use for the default values in the launching script.",
    )

    parser.add_argument(
        "--remote_config_file",
        default=None,
        help="The config file to use for the default values in the launching script on the remote machine.",
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


def load_config_from_file(config_file):
    if not os.path.isfile(config_file):
        raise FileNotFoundError(
            f"The passed configuration file `{config_file}` does not exist. "
            "Please pass an existing file to `accelerate launch`, or use the default one "
            "created through `accelerate config` and run `accelerate launch` "
            "without the `--config_file` argument."
        )

    if config_file.endswith(".json"):
        return SageMakerConfig.from_json_file(json_file=config_file)
    else:
        return SageMakerConfig.from_yaml_file(yaml_file=config_file)


def launch_command(args):
    sagemaker_config = load_config_from_file(args.config_file)

    # configure environment
    print("Configuring Amazon SageMaker environment")
    os.environ["AWS_DEFAULT_REGION"] = sagemaker_config.region

    # configure credentials
    if sagemaker_config.profile is not None:
        os.environ["AWS_PROFILE"] = sagemaker_config.profile
    elif args.aws_access_key_id is not None and args.aws_secret_access_key is not None:
        os.environ["AWS_ACCESS_KEY_ID"] = args.aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = args.aws_secret_access_key
    else:
        raise OSError("You need to provide an aws_access_key_id and aws_secret_access_key when not using aws_profile")

    # extract needed arguments
    # TODO: use source_dir but respect the .gitignore
    source_dir = os.path.dirname(args.training_script)
    if not source_dir:  # checks if string is empty
        source_dir = "."
    entry_point = os.path.basename(args.training_script)
    if not entry_point.endswith(".py"):
        raise ValueError(f'Your training script should be a python script and not "{entry_point}"')

    print("Converting Arguments to Hyperparameters")
    hyperparameters = _convert_nargs_to_dict(args.training_script_args)

    # configure sagemaker inputs
    sagemaker_inputs = None
    if sagemaker_config.sagemaker_inputs_file is not None:
        print(f"Loading SageMaker Inputs from {sagemaker_config.sagemaker_inputs_file} file")
        sagemaker_inputs = {}
        with open(sagemaker_config.sagemaker_inputs_file) as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue
                l = line.split("\t")
                sagemaker_inputs[l[0]] = l[1].strip()
        print(f"Loaded SageMaker Inputs: {sagemaker_inputs}")

    # configure sagemaker metrics
    sagemaker_metrics = None
    if sagemaker_config.sagemaker_metrics_file is not None:
        print(f"Loading SageMaker Metrics from {sagemaker_config.sagemaker_metrics_file} file")
        sagemaker_metrics = []
        with open(sagemaker_config.sagemaker_metrics_file) as file:
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

    # configure session
    print("Creating Estimator")
    args = {
        "image_uri": sagemaker_config.image_uri,
        "entry_point": entry_point,
        "source_dir": source_dir,
        "role": sagemaker_config.iam_role_name,
        "transformers_version": sagemaker_config.transformers_version,
        "pytorch_version": sagemaker_config.pytorch_version,
        "py_version": sagemaker_config.py_version,
        "base_job_name": sagemaker_config.base_job_name,
        "instance_count": sagemaker_config.num_machines,
        "instance_type": sagemaker_config.ec2_instance_type,
        "debugger_hook_config": False,
        # "distribution": distribution,
        "hyperparameters": hyperparameters,
        # "environment": environment,
        "metric_definitions": sagemaker_metrics,
    }

    if sagemaker_config.additional_args is not None:
        args = merge_dicts(sagemaker_config.additional_args, args)

    print(json.dumps(args))
    huggingface_estimator = HuggingFace(**args)

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
