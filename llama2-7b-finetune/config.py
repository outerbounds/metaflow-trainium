from dataclasses import dataclass, field
from typing import Dict, Optional
import tempfile
import os
from metaflow import IncludeFile, Parameter, JSONType


### DATASET ###
@dataclass
class DataStoreConfig:
    hf_dataset_name: str = "databricks/databricks-dolly-15k"
    hf_dataset_split: str = "train"
    local_path: str = "data/databricks-dolly-15k"
    s3_prefix: str = "databricks-dolly-15k"


### TOKENIZER ###
@dataclass
class TokenizerStoreConfig:
    local_path: str = "tokenizer.model"
    s3_prefix: str = "philschmid/Llama-2-7b-hf/tokenizer"


### MODEL ###
@dataclass
class ModelStoreConfig:
    hf_model_name: str = "philschmid/Llama-2-7b-hf"
    local_weights_path: str = "model"
    local_checkpoints_path: str = "model/checkpoints"
    s3_prefix: str = "philschmid/Llama-2-7b-hf/model"
    s3_checkpoints_key: str = "checkpoints"
    s3_experiments_key: str = "experiments"


@dataclass
class ModelArchitectureConfig:
    architectures: list = field(default_factory=lambda: ["LlamaForCausalLM"])
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 2048
    model_type: str = "llama"
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    pad_token_id: int = 0
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-05
    rope_scaling: Optional[str] = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "float16"
    transformers_version: str = "4.31.0"
    use_cache: bool = True
    vocab_size: int = 32000
    sequence_parallel_enabled: bool = False
    selective_checkpoint_enabled: bool = False
    move_model_to_device: bool = True


@dataclass
class TrainingConfig:
    tensor_parallelism_degree: int = 8 # NOTE: always keep this lower than num devices per node.
    use_mix_precision: bool = True
    use_zero_1: bool = True  # NOTE: 0 --> pure data parallelism, 1 --> ZeRO-1
    global_batch_size: int = 1024
    micro_batch_size: int = 1
    learning_rate: float = 3.0e-4
    sequence_length: int = 4096
    do_pre_compilation: bool = True
    pre_compilation_steps: int = 1
    warmup_steps: int = 3
    steps_this_run: int = 100
    total_steps: int = 100
    logging_interval: int = 1  # affects TensorBoard & CLI
    checkpoint_frequency: int = 50
    metrics_file: str = "metrics.json"


@dataclass
class TrainiumLlama2FinetuneConfig:
    data_store: DataStoreConfig = field(default_factory=DataStoreConfig)
    tokenizer_store: TokenizerStoreConfig = field(default_factory=TokenizerStoreConfig)
    model_store: ModelStoreConfig = field(default_factory=ModelStoreConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model_architecture: ModelArchitectureConfig = field(
        default_factory=ModelArchitectureConfig
    )


### ENVIRONMENT ###

# for @step cache_dataset in flow.py
caching_env_config = {
    "transformers": "4.31.0",
    "regex": "2023.12.25",
    "datasets": "2.16.1",
    "sentencepiece": "0.1.99",
    "protobuf": "3.20.0",
    "omegaconf": "2.3.0", 
}


@dataclass
class CachingEnvironmentConfig:
    batch_enabled: bool = False # NOTE: Turn this on to tokenize data remotely.
    packages: Dict[str, str] = field(default_factory=lambda: caching_env_config)


# for @step tune_llama2 in flow.py
training_env_config = {
    "optimum[neuron]": "1.16.2",
    "transformers": "4.31.0",
    "regex": "2023.12.25",
    "tensorboard": "2.15.1",
    "datasets": "2.16.1",
    "sentencepiece": "0.1.99",
    "protobuf": "3.20.0",
    "omegaconf": "2.3.0", 
}


# Derived from: https://github.com/aws-neuron/neuronx-distributed/blob/main/examples/training/llama2/tp_zero1_llama2_7b_hf_pretrain/tp_zero1_llama2_7b_hf_pretrain.sh
NUM_RT_NEURON_CORES = 32  # trn1.32xlarge instance property.
env_vars_config = {
    # "FI_EFA_USE_DEVICE_RDMA": "1",
    # "FI_PROVIDER": "efa",
    # "FI_EFA_FORK_SAFE": "1",
    # "CCOM_SOCKET_IFNAME": "eth0",
    "MALLOC_ARENA_MAX": "64",  # host OOM
    # "XLA_USE_BF16": "1",
    # "TF_NUM_INTEROP_THREADS": "8192",
    # "PROCESSES_PER_NODE": "32",
    # "NEURON_CC_FLAGS": "--model-type transformer --distribution-strategy=llm-training --cache_dir=~/neuron_compile_cache/",
    # "NEURON_FUSE_SOFTMAX": "1",
    # "NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS": "3",  # Controls number of asynchronous execution requests to be supported. Reduces latency.
    # "NEURON_RT_NUM_CORES": str(NUM_RT_NEURON_CORES),
    # "NUM_NEURONCORES": str(NUM_RT_NEURON_CORES),
    # "TPU_NUM_DEVICES": str(NUM_RT_NEURON_CORES),
    # "TPU_CHIPS_PER_HOST_BOUNDS": str(NUM_RT_NEURON_CORES),
    # "NEURON_RT_ROOT_COMM_ID": "localhost:48620",
}


@dataclass
class BatchJobConfig:
    n_nodes: int = 1
    n_trainium_devices: int = 16
    n_cpu: int = 96
    memory: int = 500000
    n_efa_interfaces: int = 8
    image: str = "public.ecr.aws/outerbounds/trainium:llama2"
    job_queue: str = "oleg2-mztdpcvj-efa"


@dataclass
class TuneLlama2EnvConfig:
    packages: Dict[str, str] = field(default_factory=lambda: training_env_config)
    env_vars: Dict[str, str] = field(default_factory=lambda: env_vars_config)
    batch_job: BatchJobConfig = field(default_factory=BatchJobConfig)
    continue_from_checkpoint_instructions: str = (
        "To continue from a checkpoint, specify the checkpoint name in the --checkpoint parameter."
    )


@dataclass
class EnvironmentConfig:
    dataset_cache_step: CachingEnvironmentConfig = field(
        default_factory=CachingEnvironmentConfig
    )
    train_llama2_step: TuneLlama2EnvConfig = field(
        default_factory=TuneLlama2EnvConfig
    )


### CONFIG HELPERS ###
def create_config(filepath, _class):
    from omegaconf import OmegaConf

    conf = OmegaConf.structured(_class)
    OmegaConf.save(conf, filepath)


def load_config(filepath, _class):
    from omegaconf import OmegaConf

    conf = OmegaConf.load(filepath)
    schema = OmegaConf.structured(_class)
    trainconf = OmegaConf.merge(schema, conf)
    return trainconf


def _to_file(file_bytes, extension=None):
    params = {
        "suffix": f".{extension.replace('.', '')}" if extension is not None else None,
        "delete": True,
        "dir": "./",
    }
    latent_temp = tempfile.NamedTemporaryFile(**params)
    latent_temp.write(file_bytes)
    latent_temp.seek(0)
    return latent_temp


class ConfigBase:
    """
    Base class for all config needed for this flow as well as any dependent flows.

    This class can be inherited by downstream classes or even used a mixin.

    This class is meant for reuse in Metaflow flows which want to resue the configuration parameters of this training flow so
    that they can call downstream flows with the same configuration parameters.

    Example Usecases:
    --------
    - Upstream flow which is preparing data is inheriting the configuration schema / parameters from this class
    - This way correct configuration parsed in both flows while we can also pass the configuration from the upstream flow to the downstream flow while ensuring that the configuration is valid.
    - This pattern is very useful when we have a complex configuration schema and we want to reuse it in multiple flows. These flows may be invoked asynchronously using event handlers, so having a common configuration schema parser is useful.

    All downstream flows will have to inherit this class and set the `config` property in this class.
    This way we will be able to access the config directly.
    The `_CORE_CONFIG_CLASS` property of this class should be set to the class which will be used to parse the configuration.

    Usage Example:
    --------
    ```
    _CORE_CONFIG_CLASS = TrainiumLlama2FinetuneConfig
    @property
    def config(self) -> TrainiumLlama2FinetuneConfig:
        return self._get_config()
    ```
    """

    def _resolve_config(self):
        if self._CORE_CONFIG_CLASS is None:
            raise ValueError(
                "Please set the _CORE_CONFIG_CLASS property of this class to the class which will be used to parse the configuration"
            )
        if (
            self.experiment_config is not None
            and self.experiment_config_file is not None
        ):
            raise ValueError("Cannot specify both --config or --config-file")
        elif self.experiment_config is None and self.experiment_config_file is None:
            raise ValueError("Must specify either --config or --config-file")
        if self.experiment_config is not None:
            return load_config(self.experiment_config, self._CORE_CONFIG_CLASS)
        if self.experiment_config_file is not None:
            temf = _to_file(
                bytes(self.experiment_config_file, "utf-8"),
            )
            return load_config(temf.name, self._CORE_CONFIG_CLASS)

    _config = None

    _CORE_CONFIG_CLASS = None

    def _get_config(self):
        if self._config is not None:
            return self._config
        self._config = self._resolve_config()
        return self._config

    experiment_config_file = IncludeFile(
        "config-file", help="experiment config file path", default=None
    )

    experiment_config = Parameter(
        "config", help="experiment config", default=None, type=JSONType
    )

    def config_report(self):
        from metaflow.cards import Markdown
        from omegaconf import OmegaConf

        return [
            Markdown(f"## Experiment Config"),
            Markdown(f"```\n{OmegaConf.to_yaml(self.config)}```"),
        ]


if __name__ == "__main__":
    if os.path.exists("config.yaml"):
        user_input = input(
            "config.yaml already exists. Type 'y/Y' and enter to overwrite: "
        ).upper()[0]
        if user_input != "Y":
            sys.exit("Exiting...")
    create_config("config.yaml", TrainiumLlama2FinetuneConfig)
