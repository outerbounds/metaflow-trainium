# Environment setup

## 🚧 Deploy Metaflow stack with AWS Batch dependencies
> Skip to [making a Docker image](#🐳-make-the-docker-image-for-training) if you already completed this section for the all reduce test and want to use the same Batch compute environment via Metaflow.

You need to install Metaflow with an AWS Batch compute environment that allows [EFA](https://aws.amazon.com/hpc/efa/) network interfaces. 
On the Neuron device side, Metaflow takes care of this.

> Follow this Terraform template. 

## Create an ECR repo for your Neuron-enabled Docker image
Login to the AWS console and create a new ECR repo called `metaflow_trn1` in your desired region.

## 🐳 Make the Docker image for training
The Docker image should be built using an x86_64-based Linux EC2 instance (ex: a c5.xlarge running Amazon Linux)
- Launch and login to your EC2 instance
- Install and start Docker
```
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user && newgrp docker
```
- Install AWS CLI (Note: this comes pre-installed on Amazon Linux)
- Run `aws configure` to add your AWS credentials to the instance (or attach a suitable IAM role to the instance)
- Git clone this repo to your instance
- `cd metaflow-trainium/llama2-7b-pretrain-trn/docker`
- From the `./docker` directory, run the following commands to build the Docker image and push it to your ECR repo 
```
export AWS_ACCT=123412341234   # <- replace with your AWS account number
export REGION=us-west-2        # <- replace with your desired AWS region
./login_ecr.sh
docker build . -t ${AWS_ACCT}.dkr.ecr.${REGION}.amazonaws.com/metaflow_trn1:latest
docker push ${AWS_ACCT}.dkr.ecr.${REGION}.amazonaws.com/metaflow_trn1:latest 
```
- In `config.py`, change the docker image in the `BatchJobConfig` to match your image's location in ECR. Also update the Job Queue to your desired trn1 job queue in AWS Batch.

# Developing

## ⚙️ Configure the run
Look at the options in `config.py` to familiarize with the project.
When satisfied, run `python config.py` and it will generate a `.yaml` file called `config.yaml`.

## ▶️ Run the flow
### Training from scratch
```
python flow.py run --config-file config.yaml
```

### Resuming from a checkpoint
At the end of each run, the workflow in `flow.py` writes the latest checkpoint to the `model_store` contained in S3. 
You can change how these results are indexed by reading the `flow.py` code if you wish. 
By default, the checkpoints are indexed in S3 by the workflow `run_id` property. 

# Limitations & workarounds
[AWS Batch Multinode](https://docs.aws.amazon.com/batch/latest/userguide/multi-node-parallel-jobs.html) jobs are not supported on step functions.
