# Environment setup

## 🚧 Deploy Metaflow stack with AWS Batch dependencies
You need to install Metaflow with an AWS Batch compute environment that allows [EFA](https://aws.amazon.com/hpc/efa/) network interfaces. 
On the Neuron device side, Metaflow takes care of this. Note that currently Metaflow overloads

> Follow this Terraform template. 

## 🐳 Make the Docker image for training
- Make `x86_64`-based EC2 image, we suggest using a `C5.xlarge` instance.
- Install Docker
- Install AWS CLI
- From the `./docker` directory, make the docker image `docker build . -t trainium`.
    - Log in to AWS ECR `aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $BASE_IMAGE_REPO` to pull the base image.
- Go to the ECR registry in your AWS account where you want to push the image to ECR, login from the EC2 instance, and push the image you just made with whatever tag you'd like.


# All reduce - distributed training infrastructure smoke test
Running this flow will test that you can create `N_NODES` Batch workers that can pass data over EFA network interfaces. 

Run the workflow:
```
python flow.py --package-suffixes=.sh run
```