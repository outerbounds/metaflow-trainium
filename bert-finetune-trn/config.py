from dataclasses import dataclass, field
from typing import Dict, Optional
import tempfile
import os
from metaflow import IncludeFile, Parameter, JSONType


### DATASET ###
@dataclass
class DataStoreConfig:
    hf_dataset_name: str = "philschmid/emotion"
    hf_dataset_split: str = "train"
    local_path: str = "data/twitter-emotion"
    s3_prefix: str = "twitter-emotion"


### TOKENIZER ###
@dataclass
class TokenizerStoreConfig:
    local_path: str = "tokenizer.model"
    s3_prefix: str = "bert-base-uncased/tokenizer"


### MODEL ###
@dataclass
class ModelStoreConfig:
    hf_model_name: str = "bert-base-uncased"
    local_weights_path: str = "model"
    local_checkpoints_path: str = "model/checkpoints"
    s3_prefix: str = "bert-base-uncased/model"
    s3_checkpoints_key: str = "checkpoints"
    s3_experiments_key: str = "experiments"


@dataclass
class TrainingConfig:
    bf16: bool = True
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 16
    tensor_parallel_size: int = 1
    epochs: int = 3
    logging_steps: int = 1
    gradient_accumulation_steps: int = 1


@dataclass
class TrainiumBERTFinetuneConfig:
    data_store: DataStoreConfig = field(default_factory=DataStoreConfig)
    tokenizer_store: TokenizerStoreConfig = field(default_factory=TokenizerStoreConfig)
    model_store: ModelStoreConfig = field(default_factory=ModelStoreConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


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
    batch_enabled: bool = False  # NOTE: Turn this on to tokenize data remotely.
    packages: Dict[str, str] = field(default_factory=lambda: caching_env_config)


# for @step tune_bert in flow.py
training_env_config = {
    "git+https://github.com/huggingface/optimum-neuron.git": "",
    "evaluate": "0.4.1",
    "regex": "2023.12.25",
    # "tensorboard": "2.15.1",
    "datasets": "2.16.1",
    "sentencepiece": "0.1.99",
    "protobuf": "3.20.0",
    "omegaconf": "2.3.0",
}


env_vars_config = {
    # "FI_EFA_USE_DEVICE_RDMA": "1",
    # "FI_PROVIDER": "efa",
    "FI_EFA_FORK_SAFE": "1",
    "CCOM_SOCKET_IFNAME": "eth0",
    "MALLOC_ARENA_MAX": "64",  # host OOM
}


# trn1.2xlarge
class BatchJobConfig:
    n_nodes: int = 1
    n_trainium: int = 1
    n_cpu: int = 4
    memory: int = 24000
    image: str = "public.ecr.aws/outerbounds/trainium:llama2"
    job_queue: str = "small-trainium"
    use_tmpfs: bool = False


@dataclass
class TuneBERTEnvConfig:
    packages: Dict[str, str] = field(default_factory=lambda: training_env_config)
    env_vars: Dict[str, str] = field(default_factory=lambda: env_vars_config)
    batch_job: BatchJobConfig = field(default_factory=BatchJobConfig)


@dataclass
class EnvironmentConfig:
    dataset_cache_step: CachingEnvironmentConfig = field(
        default_factory=CachingEnvironmentConfig
    )
    tune_bert_step: TuneBERTEnvConfig = field(default_factory=TuneBERTEnvConfig)


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
    _CORE_CONFIG_CLASS = TrainiumBERTFinetuneConfig
    @property
    def config(self) -> TrainiumBERTFinetuneConfig:
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
    create_config("config.yaml", TrainiumBERTFinetuneConfig)
