#!/bin/bash

if [ ! -v AWS_ACCT ] || [ ! -v REGION ]
then
  echo "Error: Please export AWS_ACCT AND REGION environment variables and try again. Ex:"
  echo "  export AWS_ACCT=123412341234"
  echo "  export REGION=us-west-2"
  echo "  $0"
  exit 1
fi

MY_ECR_REPO_URI=${AWS_ACCT}.dkr.ecr.${REGION}.amazonaws.com/metaflow_trn1
DLC_ECR_REPO_URI=763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training-neuronx

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$MY_ECR_REPO_URI"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$DLC_ECR_REPO_URI"
