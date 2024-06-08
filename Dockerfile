FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

 # For conda ssl verification
# ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# RUN curl -L -o ~/miniforge3.sh https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE3_VERSION}/Miniforge3-${MINIFORGE3_VERSION}-Linux-x86_64.sh \
#  && chmod +x ~/miniforge3.sh \
#  && ~/miniforge3.sh -b -p /opt/conda \
#  && rm ~/miniforge3.sh

#RUN pip install --upgrade pip --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org

# Install SM packages
RUN pip install --no-cache-dir -U \
    smclarify \
    "sagemaker>=2,<3" \
    "sagemaker-experiments<1" \
    sagemaker-pytorch-training \
    sagemaker-training

# Copy workaround script for incorrect hostname
COPY docker/changehostname.c /
COPY docker/start_with_right_hostname.sh /usr/local/bin/start_with_right_hostname.sh

RUN chmod +x /usr/local/bin/start_with_right_hostname.sh

# Removing the cache as it is needed for security verification
RUN rm -rf /root/.cache | true

ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
CMD ["/bin/bash"]