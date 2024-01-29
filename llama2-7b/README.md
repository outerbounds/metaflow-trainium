# Environment setup

## 🚧 Deploy Metaflow stack with AWS Batch dependencies
> Skip to [making a Docker image](#🐳-make-the-docker-image-for-training) if you already completed this section for the all reduce test and want to use the same Batch compute environment via Metaflow.

You need to install Metaflow with an AWS Batch compute environment that allows [EFA](https://aws.amazon.com/hpc/efa/) network interfaces. 
On the Neuron device side, Metaflow takes care of this.

> Follow this Terraform template. 

## 🐳 Make the Docker image for training
- Make `x86_64`-based EC2 image, we suggest using a `C5.xlarge` instance.
- Install Docker
- Install AWS CLI
- From the `./docker` directory, make the docker image `docker build . -t trainium`.
    - Log in to AWS ECR `aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $BASE_IMAGE_REPO` to pull the base image.
- Go to the ECR registry in your AWS account where you want to push the image to ECR, login from the EC2 instance, and push the image you just made with whatever tag you'd like.
- In `config.py`, change the docker image in the `BatchJobConfig` to match your image's location.

# Developing

## ⚙️ Configure the run
Look at the options in `config.py` to familiarize with the project.
When satisfied, run `python config.py` and it will generate a `.yaml` file called `config.yaml`.

## ▶️ Run the flow
### Training from scratch
```
python flow.py run --config config.yaml
```

### Resuming from a checkpoint
At the end of each run, the workflow in `flow.py` writes the latest checkpoint to the `model_store` contained in S3. 
You can change how these results are indexed by reading the `flow.py` code if you wish. 
By default, the checkpoints are indexed in S3 by the workflow `run_id` property. 

# Limitations & workarounds
[AWS Batch Multinode](https://docs.aws.amazon.com/batch/latest/userguide/multi-node-parallel-jobs.html) jobs are not supported on step functions.