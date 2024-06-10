# Set the default value for the REGION build argument
ARG REGION=us-east-1

# SageMaker PyTorch image for TRAINING
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker

RUN pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu121
RUN pip install git+https://github.com/huggingface/transformers

RUN MAX_JOBS=4 pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# Display installed packages for reference
RUN pip freeze