import boto3
import argparse
import json
import os

MF_CONFIG_VARS_TO_CFN_OUTPUT_KEYS = {
    "METAFLOW_BATCH_JOB_QUEUE": "BatchJobQueueArn",
    # 'METAFLOW_SERVICE_INTERNAL_URL': 'InternalServiceUrl',
    "METAFLOW_SFN_DYNAMO_DB_TABLE": "DDBTableName",
    "METAFLOW_DATATOOLS_S3ROOT": "MetaflowDataToolsS3Url",
    "METAFLOW_SFN_IAM_ROLE": "StepFunctionsRoleArn",
    "METAFLOW_SERVICE_URL": "ServiceUrl",
    "METAFLOW_DATASTORE_SYSROOT_S3": "MetaflowDataStoreS3Url",
    "METAFLOW_ECS_S3_ACCESS_IAM_ROLE": "ECSJobRoleForBatchJobs",
    "METAFLOW_EVENTS_SFN_ACCESS_IAM_ROLE": "EventBridgeRoleArn",
}
EXTRA_ARGS = {
    "METAFLOW_DEFAULT_DATASTORE": "s3",
    "METAFLOW_DEFAULT_METADATA": "service",
}


def fetch_config_vars(stack_name: str) -> dict:
    client = boto3.client("cloudformation")
    response = client.describe_stacks(StackName=stack_name)
    outputs = response["Stacks"][0]["Outputs"]
    outputs_dict = {output["OutputKey"]: output["OutputValue"] for output in outputs}
    cfg_dict = {
        cfg_var: outputs_dict[cfn_output_key]
        for cfg_var, cfn_output_key in MF_CONFIG_VARS_TO_CFN_OUTPUT_KEYS.items()
    }
    cfg_dict |= EXTRA_ARGS
    return cfg_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stack_name", type=str, required=True)
    parser.add_argument("-p", "--profile", type=str, default="default")
    parser.add_argument(
        "-home",
        "--metaflow-home",
        type=str,
        default=os.path.expanduser("~/.metaflowconfig"),
    )
    args = parser.parse_args()
    mf_cfg = fetch_config_vars(args.stack_name)
    if not os.path.exists(args.metaflow_home):
        os.makedirs(args.metaflow_home)
    if args.profile == "default":
        config_path = os.path.join(args.metaflow_home, "config.json")
    else:
        config_path = os.path.join(args.metaflow_home, f"config_{args.profile}.json")
    with open(config_path, "w") as f:
        json.dump(mf_cfg, f, indent=4)
    print(f"Config written to {config_path}")
