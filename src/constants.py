SUPPORTED_MODELS = ["TinyLlama-1.1B", "Meta-Llama-3-8B"]
SM_MODEL_DIR = "/opt/ml/model"
SM_TENSORBOARD_OUTPUT_DIRECTORY = "/opt/tensorboard"
SAGEMAKER_CODE_TAR_GZ = "code.tar.gz"

SAGEMAKER_SUPPORTED_INSTANCES_AND_GPUS = {
    "ml.g5.xlarge": 1,
    "ml.g5.2xlarge": 1,
    "ml.g5.4xlarge": 1,
    "ml.g5.8xlarge": 1,
    "ml.g5.16xlarge": 1,
    "ml.g5.12xlarge": 4,
    "ml.g5.24xlarge": 4,
    "ml.g5.48xlarge": 8,
}
