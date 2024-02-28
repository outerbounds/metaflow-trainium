# Allreduce - distributed training infrastructure smoke test
Running this flow will test that you can create `N_NODES` Batch workers that can pass data over EFA network interfaces. 

First, modify `flow.py` and update the `image=` and `queue=` lines to use your Trainium Docker image and Batch queue, respectively. If you are unsure of these values, please refer to the `Outputs` tab for the AWS Batch resources stack that was created as part of the [installation instructions](../install_metaflow_and_batch.md).
 
Run the workflow:
```
python flow.py --package-suffixes=.sh run
```
