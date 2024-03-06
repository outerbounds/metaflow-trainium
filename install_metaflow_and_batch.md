# Deploying Metaflow and AWS Batch resources

This section explains how to deploy the required Metaflow and AWS Batch resources in AWS, and configure Metaflow to enable Trainium-based training jobs.

AWS Trainium is currently supported in us-east-1, us-east-2, and us-west-2. In the following steps, please make sure that you are working in one of these supported regions.

## 🛠️ Deploy Metaflow stack in AWS
If you have not previously deployed the Metaflow stack in AWS, please follow these steps to download and deploy the CloudFormation template:
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
- When the stack has been deployed, go to the `Outputs` tab. This tab shows the values that you will need when configuring Metaflow on your instance in subsequent steps

Please note that this default Metaflow stack will create a Batch job queue that is not enabled for Trainium / trn1 instances. The Trainium-enabled job queue will created in the next section, and it is the queue that should be specified when launching Trainium-based training jobs. 

## 🛠️ Create AWS Batch resources
Before you can run AWS Trainium jobs in AWS Batch, you first need to create a VPC with Trainium-supported subnets, EC2 launch template, AWS Batch compute environment, and AWS Batch job queue. If you have not yet created these resources, you can use the provided Cloudformation template to quickly deploy a basic setup to get you started.

You can either download the [Cloudformation template](./cfn/trn1_batch_resources.yaml) and deploy the stack in the AWS console, or deploy the stack from the command-line as follows:
```
REGION=us-west-2       # replace with your desired region
STACKNAME=trn1-batch   # replace with your desired stack name

aws cloudformation --region $REGION create-stack \
--stack-name $STACKNAME \
--template-body file://cfn/trn1_batch_resources.yaml \
--capabilities CAPABILITY_IAM
```

You can monitor the stack deployment process in the AWS CloudFormation console. Once the stack has deployed, look at the `Outputs` tab to determine the name of your AWS Batch job queue an the URI for your ECR repository. You will need these values in subsequent steps.

## 🐳 Make the Docker image for training
The Docker image should be built using an x86_64-based Linux EC2 instance (ex: a c5.xlarge running Amazon Linux)
- Launch and login to your EC2 instance
- Install and start Docker, ex:
```
sudo yum install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user && newgrp docker
```
- Install AWS CLI (Note: this comes pre-installed on Amazon Linux)
- Run `aws configure` to add your AWS credentials to the instance (or attach a suitable IAM role to the instance)
- Git clone this repo to your instance
```
git clone https://github.com/metaflow-trainium.git
cd metaflow-trainium
```
- Run the following commands to build the Docker image and push it to your ECR repo. You can determine the URI of your ECR repo by referring to the `Outputs` tab for your AWS Batch Cloudformation stack that you deployed, above. 
```
MY_ECR_REPO=123412341234.dkr.ecr.us-west-2.amazonaws.com/test-metaflow_trn1  # <- replace with your ECR URI
REGION=us-west-2                                                             # <- replace with your desired AWS region
DLC_ECR_REPO=763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training-neuronx

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$MY_ECR_REPO"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$DLC_ECR_REPO"

docker build ./docker -t $MY_ECR_REPO:latest
docker push $MY_ECR_REPO:latest
```

## 🛠️ Install and configure Metaflow
- First create a new virtual environment and install Metaflow & related packages:
```
python3 -m venv metaflow_venv
. ./metaflow_venv/bin/activate
pip3 install -U pip
pip3 install metaflow omegaconf
pip3 install git+https://github.com/outerbounds/metaflow-torchrun.git@dff2b73c0251919f84c2ebb0ece6475b8d9bd0a9 
```
- Next, run `metaflow configure aws`. When prompted, enter the appropriate values from the Metaflow CloudFormation stack's `Outputs` tab.
**Note:** please skip the optional `METAFLOW_SERVICE_INTERNAL_URL` value, as it will cause issues if your Metaflow resources and Batch resources use different VPCs.

---

Congratulations! You can now try running one of the Metaflow-Trainium examples. The [allreduce example](./allreduce-trn) is a great place to start.
