# Llama2-7B pretraining

This flow will run a multi-node Llama2-7B pretraining job in AWS Batch using AWS EC2 trn1 instances with AWS Trainium ML chips.

Note: this example uses the Meta Llama2 model which is bound by [Meta's Llama license](https://llama.meta.com/llama-downloads/). Please ensure that you have agreed to the license terms and requested access to the Llama model files, then copy the `config.json` and `tokenizer.model` files from the [Hugging Face Llama repo](https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main) to the `lama2-7b-pretrain-trn` example directory before proceeding.
`
## ⚙️ Configure the run
Look at the options in `config.py` to familiarize yourself with the project.
In `config.py`, change the Docker image in the `BatchJobConfig` to match your image's location in ECR. Also update the Job Queue to your desired trn1 job queue in AWS Batch.
When satisfied, run `python config.py` and it will generate a `.yaml` file called `config.yaml`.

## ▶️ Run the flow
### Training from scratch
```
python flow.py run --config-file config.yaml
```

### Monitoring
To observe results, you can either deploy the [Metaflow UI](https://github.com/Netflix/metaflow-ui), or use the local card server, which is built-in.
To run the local server, type the following command in the same virtual environment where the `flow.py` is running:
```
python flow.py card server 
```
The open https://localhost:8342 and you can observe the real-time updates from your multi-node batch job.

### Resuming from a checkpoint
At the end of each run, the workflow in `flow.py` writes the latest checkpoint to the `model_store` contained in S3. 
You can change how these results are indexed by reading the `flow.py` code if you wish. 
By default, the checkpoints are indexed in S3 by the workflow `run_id` property. 
