# Metaflow-Trainium Examples

This repository contains examples that demonstrate how to use [Metaflow](https://metaflow.org/) to define and run machine learning training jobs with [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/). The training jobs are executed as batch jobs running on AWS EC2 trn1 instances in [AWS Batch](https://aws.amazon.com/batch/).

To run these examples, you first need to provision AWS resources for Metaflow and AWS Batch. Please refer to the [installation guide](./install_metaflow_and_batch.md) for instructions on how to deploy the required resources using CloudFormation and finalize your Metaflow setup.

Once the required resources have been created and configured, please try to run the included [allreduce example](./allreduce-trn) as a basic test of the Metaflow/Trainium/Batch setup. When the allreduce example is successfully running, you can then proceed to the more interesting examples such as [Llama2-7b pretraining](./llama2-7b-pretrain-trn).

AWS Trainium is currently supported in us-east-1, us-east-2, and us-west-2. Please make sure that you are working in one of these supported regions.

## Example registry

We have included the following examples, and are happy to take requests to expand the list. Note that some sub-directories for Trainium have counterpart implementations for running comparisons on GPUs. This is not intended to be a benchmarking repository, but running a comparison against GPUs you have access to is useful for understanding general performance characteristics relative to other hardware architectures.

### [Llama2 pre-training](./llama2-7b-pretrain-trn/)
Pre-train Llama2 using ≥4 nodes with `trn.32xlarge` instances. 

### [Llama2 fine-tuning on Trainium](./llama2-7b-finetune-trn/)
Fine-tune Llama2 on a single `trn.32xlarge` instance using the [`optimum-neuron`](https://huggingface.co/docs/optimum-neuron/en/index) library from Huggingface. 

For a minimal code change GPU implementation, see [here](./llama2-7b-finetune-gpu-single-node/). 
Note: We found A100 GPUs to have the most comparable characteristics, but it is far from an apples-to-apples comparison.

### [BERT fine-tuning on Trainium](./bert-finetune-trn/)
Fine-tune BERT on a single `trn.2xlarge` instance using the `optimum-neuron` library from Huggingface. 

For a minimal code change GPU implementation, see [here](./bert-finetune-gpu/). 