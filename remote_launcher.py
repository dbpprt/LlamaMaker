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

    sys.argv = old_sys_argv[0:2] + [training_script] + old_sys_argv[2:]
    print(f"Launching lunch command with args: {' '.join(sys.argv)}")

    try:
        return parser.parse_args()
    finally:
        sys.argv = old_sys_argv


def main():
    # training_script
    parser = launch_command_parser()
    # get training script from env var
    training_script = os.environ["ENTRYPOINT"]

    assert isinstance(training_script, str), f"{training_script} is not a string!"
    assert os.path.exists(training_script), f"{training_script} does not exist!"
    print(f"Launching training_script {training_script}")

    args = parse_args(parser, training_script="train.py")
    launch_command(args)


if __name__ == "__main__":
    main()
