<a name="readme-top"></a>

<div align="center">
  <a href="https://https://github.com/dbpprt/LeanLLM-X">
    <img src="docs/images/logo.png" alt="Logo" width="120" height="120">
  </a>
</div>

# [LlamaMaker](https://github.com/dbpprt/LlamaMaker)

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

### Build, Train, and Fine-tune Large Language Models on [Amazon SageMaker](https://aws.amazon.com/sagemaker/) 🚀

Welcome to the **LlamaMaker** repository, a easy to use solution to build and fine-tune *Large Language Models* unlocking the power of [Gen AI](https://aws.amazon.com/generative-ai/). Harness the capabilities of [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) (soon), [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) (soon) and [NVIDIA GPUs](https://aws.amazon.com/nvidia/) to scale your fine-tuning with ease.

This solution provides you an easy to use abstraction layer to fine-tune custom Llama variants locally or remotely using SageMaker training jobs. We support distributed training on **g5.**, **p4d**, and **p5** instances. LlamaMaker streams **Tensorboard** results back and allows you to easily scale your training jobs .


> **Note**: LlamaMaker is actively being developed. To see what features are in progress, please check out the [issues](https://github.com/dbpprt/LlamaMaker/issues) section of our repository.

## 🏗️ Architecture
- LamaMaker is built on top of [🤗 transformers](), [🤗 peft](), [🤗 trl](), [🤗 accelerate]() and integrates with the SageMaker SDK.
- Custom training container images with automated build pipeline (based on GitHub Action, hosted in AWS CodeBuild)
- Local first: LlamaMaker is designed to run locally on Apple Silicon, providing a first class experience for developers.

## 🌟 Features
- 🎯 Custom container support with integrated deployment and build pipeline.
- 🎯 BYOD - Bring your own datasets, models or both *without writing any code*.
- 🎯 Local first: LlamaMaker is designed to run locally on Apple Silicon, providing a first class experience for developers. (MPS backend)
- 🎯 Support for **fp32**, **fp16**, **fp8**, **QLoRa**, **LoRa** and more.
- 🎯 Tensorboard integration to monitor training progress..
- 🎯 [smdistributed](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel-modify-sdp-pt.html) on **p4d.***
- 🎯 **Single-node multi-GPU** fully supported
- 🎯 Extensive validation metrics for JSON generation (schema validation, field based accuracy, and more)
- 🎯 *coming soon*: **Multi-node multi-gpu**
- 🎯 *coming soon*: **FSDP**, **DeepSpeed**

## 🏃‍♀️Getting Started

### Setup your development environment

```bash
# make sure to have a local conda environment, otherwise
# TODO: pathes are likely not working, due to folder restructuring
chmod +x scripts/development-environment/environment.sh
sh scripts/development-environment/environment.sh
```

```bash
conda env create -f scripts/development-environment/environment.yaml
conda activate llamamaker
```

### Fine-tune locally using [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) with the [MPS backend](https://pytorch.org/docs/stable/notes/mps.html)
```bash
# note: this only works on Apple Silicon and is intended for debugging purposes!
accelerate launch --config_file=./config/local.yaml \
                    train.py \
                    --model_id TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
                    --data_config ./data/swisstext2023/llama3.yaml \
                    --debug \
                    --per_device_train_batch_size 1 \
                    --per_device_eval_batch_size 1 \
                    --gradient_accumulation_steps 1 \
                    --max_seq_length 256 \
                    --logging_steps 1 \
                    --eval_steps 5 \
                    --save_steps 50 \
                    --num_train_epochs 1 \
                    --optim "adamw_hf" \
                    --lora_modules_to_save "embed_tokens" \
                    --lora_r 64 \
                    --lora_alpha 16 \
                    --lora_dropout 0.1
```


## 🗂️ Documentation


## 🏆 Motivation

## 🤝 Support & Feedback
**LlamaMaker** is maintained by AWS Solution Architects and is not an AWS service. Support is provided on a best effort basis by the community. If you have feedback, feature ideas, or wish to report bugs, please use the [Issues](https://github.com/dbpprt/LlamaMaker/issues) section of this GitHub.

## 🔐 Security
See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## 💼 License
This library is licensed under the Apache 2.0 License.

## 🙌 Community
We welcome all individuals who are enthusiastic about machine learning to become a part of this open source community. Your contributions and participation are invaluable to the success of this project.

Built with ❤️ at AWS.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username