# Metaflow-Trainium Examples

This repository contains examples that demonstrate how to use [Metaflow](https://metaflow.org/) to define and run machine learning training jobs with [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/). The training jobs are executed as batch jobs running on AWS EC2 trn1 instances in [AWS Batch](https://aws.amazon.com/batch/).

To run these examples, you first need to provision AWS resources for Metaflow and AWS Batch. Please refer to the [installation guide](./install_metaflow_and_batch.md) for instructions on how to deploy the required resources using CloudFormation and finalize your Metaflow setup.

Once the required resources have been created and configured, please try to run the included [allreduce example](./allreduce-trn) as a basic test of the Metaflow/Trainium/Batch setup. When the allreduce example is successfully running, you can then proceed to the more interesting examples such as [Llama2-7b pretraining](./llama2-7b-pretrain-trn).

AWS Trainium is currently supported in us-east-1, us-east-2, and us-west-2. Please make sure that you are working in one of these supported regions.
