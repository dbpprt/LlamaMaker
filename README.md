![LlamaMaker](docs/logo.png)
# [LlamaMaker](https://github.com/dbpprt/LlamaMaker)

[![plan-examples](https://github.com/awslabs/data-on-eks/actions/workflows/plan-examples.yml/badge.svg?branch=main)](https://github.com/awslabs/data-on-eks/actions/workflows/plan-examples.yml)

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
- 🎯 Local first: LlamaMaker is designed to run locally on Apple Silicon, providing a first class experience for developers.
- 🎯 Support for **fp32**, **fp16**, **fp8**, **QLoRa**, **LoRa** and more.
- 🎯 Tensorboard integration to monitor training progress..
- 🎯 
- 🎯
- 🎯
- 🎯

## 🏃‍♀️Getting Started




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