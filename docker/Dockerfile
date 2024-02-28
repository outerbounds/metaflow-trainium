FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training-neuronx:1.13.1-neuronx-py310-sdk2.17.0-ubuntu20.04

WORKDIR /

# Set up from the shell script in the example
RUN ["/bin/bash", "-c", "set -euo pipefail"]
RUN ["/bin/bash", "-c", "/usr/local/bin/python -m pip install neuronx_distributed --extra-index-url https://pip.repos.neuron.amazonaws.com"]
RUN pip install --upgrade pip
RUN pip install transformers==4.31.0 regex==2023.12.25 tensorboard==2.15.1 datasets==2.16.1 sentencepiece==0.1.99 protobuf==3.20.0 omegaconf==2.3.0
# NOTE: /usr/local/bin/python is the binary Metaflow uses
RUN ["/bin/bash", "-c", "ulimit -n 65535"]
# RUN ["/bin/bash", "-c", "net.ipv4.ip_local_reserved_ports=41000"]
# NOTE: The above reserved the port 41000 for torchrun to use for the NCCL backend, which is used by neuronx_distributed. 
