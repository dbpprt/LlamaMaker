import os
import sys

from accelerate.commands.launch import launch_command, launch_command_parser


# very simple wrapper for the accelerate launch command to be used by SageMaker


# this is a wrapper to inject the training_script into the command line arguments
# SageMaker doesn't support arguments without hyphens as they're passed as hyperparameters
def parse_args(parser, training_script):
    # this is quite a hacky way to inject the training script into the arguments, but it works
    old_sys_argv = sys.argv

    print(f"remote_launcher.py called with args: {' '.join(old_sys_argv)}")

    assert old_sys_argv[1].startswith(
        "--config_file"
    ), "expected first argument to be --config_file for this launcher to work"

    # sometimes the config_file is a single arg, sometimes it is 2 different args, so we need to handle both cases
    if old_sys_argv[1].endswith(".yaml"):
        sys.argv = old_sys_argv[0:2] + [training_script] + old_sys_argv[2:]
    else:
        sys.argv = old_sys_argv[0:3] + [training_script] + old_sys_argv[3:]

    print(f"Launching lunch command with args: {' '.join(sys.argv)}")

    try:
        return parser.parse_args()
    finally:
        sys.argv = old_sys_argv


def main():
    # training_script
    parser = launch_command_parser()
    # get training script from env var
    if "ENTRYPOINT" in os.environ:
        training_script = os.environ["ENTRYPOINT"]
    else:
        print("No entrypoint given, falling back to train.py")
        training_script = "train.py"

    assert isinstance(training_script, str), f"{training_script} is not a string!"
    assert os.path.exists(training_script), f"{training_script} does not exist!"
    print(f"Launching training_script {training_script}")

    args = parse_args(parser, training_script="train.py")

    print("Launching launch command with args:", args)
    launch_command(args)


if __name__ == "__main__":
    main()
