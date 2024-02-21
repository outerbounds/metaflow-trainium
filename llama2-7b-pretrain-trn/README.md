# Environment setup

## 🚧 Deploy Metaflow stack in AWS
If you have not previous deployed the Metaflow stack in AWS, please follow these steps to download and deploy the CloudFormation template:
- Download the template [HERE](https://github.com/outerbounds/metaflow-tools/blob/master/aws/cloudformation/metaflow-cfn-template.yml)
- Open the AWS console, and search for "CloudFormation"
- Choose "Create Stack" -> "With new resources (standard)"
- Under "Prepare template" select "Template is ready"
- Under "Template source" select "Upload a template file" and then click the "Choose file" button
- In the file selector, choose the template file you downloaded above, then click "Next"
- Name your stack under "Stack name"
- Set "APIBasicAuth" to false
- Leave all other fields as their default values
- Click "Next" through the subsequent screens, then "Submit" to begin stack creation
- When the stack has been deployed, go to the "Outputs" tab. This tab shows the values that you will need when configuring Metaflow on your instance in subsequent steps

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
```
git clone https://github.com/5cp/metaflow-trainium.git -b aws_testing
cd metaflow-trainium/llama2-7b-pretrain-trn/docker
```
- From the `./docker` directory, run the following commands to build the Docker image and push it to your ECR repo 
```
export AWS_ACCT=123412341234   # <- replace with your AWS account number
export REGION=us-west-2        # <- replace with your desired AWS region
./login_ecr.sh
docker build . -t ${AWS_ACCT}.dkr.ecr.${REGION}.amazonaws.com/metaflow_trn1:latest
docker push ${AWS_ACCT}.dkr.ecr.${REGION}.amazonaws.com/metaflow_trn1:latest 
```

## Install and configure Metaflow
- First create a new virtual environment and install Metaflow & related packages:
```
python3 -m venv metaflow_venv
. ./metaflow_venv/bin/activate
pip3 install -U pip
pip3 install -r omegaconf
pip3 install git+https://github.com/outerbounds/metaflow-torchrun.git@dff2b73c0251919f84c2ebb0ece6475b8d9bd0a9 
```
- Next, run `metaflow configure aws`. When prompted, enter the appropriate values from the Metaflow CloudFormation stack's Outputs tab.
**Note:** please skip the optional `METAFLOW_SERVICE_INTERNAL_URL` value, as it will cause issues if your Metaflow resources and Batch resources use different VPCs.

## Create AWS Batch resources
TODO

# Developing

## ⚙️ Configure the run
Look at the options in `config.py` to familiarize yourself with the project.
In `config.py`, change the docker image in the `BatchJobConfig` to match your image's location in ECR. Also update the Job Queue to your desired trn1 job queue in AWS Batch.
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
