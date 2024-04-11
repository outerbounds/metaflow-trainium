from metaflow import FlowSpec, step, batch, torchrun, environment, current
import os
import sys

from config import ConfigBase, EnvironmentConfig, BERTFinetuneConfig
from ops import DataStore, TokenizerStore, ModelStore
from custom_decorators import pip, enable_decorator
from gpu_profile import gpu_profile

environment_config = EnvironmentConfig()


class BERTFinetune(FlowSpec, ConfigBase):

    _CORE_CONFIG_CLASS = BERTFinetuneConfig

    def _get_data_store(self):
        return DataStore.from_config(self.config.data_store)

    def _get_tokenizer_store(self):
        return TokenizerStore.from_config(self.config.tokenizer_store)

    def _get_model_store(self):
        return ModelStore.from_config(self.config.model_store)

    @property
    def config(self) -> BERTFinetuneConfig:
        return self._get_config()

    @pip(packages={**environment_config.dataset_cache_step.packages})
    @step
    def start(self):
        tokenizer_store = self._get_tokenizer_store()
        if not tokenizer_store.already_exists():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_store.hf_model_name
            )
            tokenizer.save_pretrained(self.config.tokenizer_store.local_path)
            tokenizer_store.upload(self.config.tokenizer_store.local_path)
        self.next(self.cache_dataset)

    @pip(packages={**environment_config.dataset_cache_step.packages})
    @step
    def cache_dataset(self):
        data_store = self._get_data_store()
        if not data_store.already_exists():
            tokenizer_store = self._get_tokenizer_store()
            tokenizer_store.download(self.config.tokenizer_store.local_path)
            data_store.download_from_huggingface(
                self.config.data_store, self.config.tokenizer_store.local_path
            )
            data_store.upload(self.config.data_store.local_path)
        self.next(
            self.tune_bert,
            num_parallel=environment_config.tune_bert_step.batch_job.n_nodes,
        )

    @gpu_profile(interval=1)
    @pip(packages={**environment_config.tune_bert_step.packages})
    @environment(vars=environment_config.tune_bert_step.env_vars)
    @batch(
        gpu=environment_config.tune_bert_step.batch_job.n_gpu,
        cpu=environment_config.tune_bert_step.batch_job.n_cpu,
        memory=environment_config.tune_bert_step.batch_job.memory,
        image=environment_config.tune_bert_step.batch_job.image,
        queue=environment_config.tune_bert_step.batch_job.queue,
    )
    @torchrun
    @step
    def tune_bert(self):

        from omegaconf import OmegaConf
        import json

        def make_path(rel_path, make_dir=True, use_tmpfs=False):
            if use_tmpfs:
                path = os.path.join(current.tempdir, rel_path)
            else:
                path = os.path.join(os.getcwd(), rel_path)
            if not os.path.exists(path) and make_dir:
                os.makedirs(path)
            return path

        data_dir = make_path(self.config.data_store.local_path)
        checkpoint_dir = make_path(
            self.config.model_store.local_checkpoints_path, use_tmpfs=False
        )
        model_path = make_path(self.config.model_store.local_weights_path)

        # Download tokenized data.
        data_store = self._get_data_store()
        data_store.download(download_path=data_dir)

        # convert deepspeed config to json file
        deepspeed_config_file = "ds_config.json"
        model_arch_config = OmegaConf.to_container(self.config.deepspeed)
        with open(os.path.join(model_path, deepspeed_config_file), "w") as f:
            json.dump(model_arch_config, f)

        entrypoint_args = {
            "model_id": self.config.model_store.hf_model_name,
            "dataset_path": data_dir,
            "pretrained_model_cache": os.path.join(
                current.tempdir, "pretrained_model_cache"
            ),
            "bf16": self.config.training.bf16,
            "lr": self.config.training.learning_rate,
            "output_dir": checkpoint_dir,
            "per_device_train_batch_size": self.config.training.per_device_train_batch_size,
            "epochs": self.config.training.epochs,
            "logging_steps": self.config.training.logging_steps,
            "deepspeed": deepspeed_config_file,
        }

        # Train the model.
        current.torch.run(
            entrypoint="train.py", entrypoint_args=entrypoint_args, master_port="41000"
        )
        model_store = self._get_model_store()
        model_store.upload(
            local_path=checkpoint_dir,
            store_key=os.path.join(
                self.config.model_store.s3_checkpoints_key,
                current.run_id,
                str(current.parallel.node_index),
            ),
        )

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    BERTFinetune()
