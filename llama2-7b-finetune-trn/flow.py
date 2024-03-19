from metaflow import FlowSpec, step, batch, torchrun, environment, current
import os
import sys

from config import ConfigBase, EnvironmentConfig, TrainiumLlama2FinetuneConfig
from ops import DataStore, TokenizerStore, ModelStore
from custom_decorators import pip, enable_decorator
from neuron_monitor import neuron_monitor

environment_config = EnvironmentConfig()


class TrainiumLlama2Finetune(FlowSpec, ConfigBase):

    _CORE_CONFIG_CLASS = TrainiumLlama2FinetuneConfig

    def _get_data_store(self):
        return DataStore.from_config(self.config.data_store)

    def _get_tokenizer_store(self):
        return TokenizerStore.from_config(self.config.tokenizer_store)

    def _get_model_store(self):
        return ModelStore.from_config(self.config.model_store)

    @property
    def config(self) -> TrainiumLlama2FinetuneConfig:
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
    @enable_decorator(batch, environment_config.dataset_cache_step.batch_enabled)
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
            self.tune_llama2,
            num_parallel=environment_config.tune_llama2_step.batch_job.n_nodes,
        )

    @pip(packages={**environment_config.tune_llama2_step.packages})
    @environment(vars=environment_config.tune_llama2_step.env_vars)
    @neuron_monitor(interval=1)
    @batch(
        trainium=environment_config.tune_llama2_step.batch_job.n_trainium_devices,
        efa=environment_config.tune_llama2_step.batch_job.n_efa_interfaces,
        cpu=environment_config.tune_llama2_step.batch_job.n_cpu,
        memory=environment_config.tune_llama2_step.batch_job.memory,
        image=environment_config.tune_llama2_step.batch_job.image,
        queue=environment_config.tune_llama2_step.batch_job.job_queue,
        use_tmpfs=True,  # size is 1/2 of `memory` by default.
    )
    @torchrun
    @step
    def tune_llama2(self):

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
            self.config.model_store.local_checkpoints_path, use_tmpfs=True
        )

        # Download tokenized data.
        data_store = self._get_data_store()
        data_store.download(download_path=data_dir)

        # Download the neuron compiler cache.
        neuron_compiler_cache_dir = "/var/tmp/neuron-compile-cache"
        try:
            model_store = self._get_model_store()
            model_store.download(
                download_path=neuron_compiler_cache_dir,
                store_key=self.config.model_store.s3_neuron_compiler_cache_key
            )
        except ValueError as e:
            print('Compiler cache is empty, optimum trainer will tell neuron-cc to compile the model. It might take a while...')

        entrypoint_args = {
            "model_id": self.config.model_store.hf_model_name,
            "dataset_path": data_dir,
            "bf16": self.config.training.bf16,
            "learning_rate": self.config.training.learning_rate,
            "output_dir": checkpoint_dir,
            "overwrite_output_dir": self.config.training.overwrite_output_dir,
            "skip_cache_push": self.config.training.skip_cache_push,
            "per_device_train_batch_size": self.config.training.per_device_train_batch_size,
            "gradient_checkpointing": self.config.training.gradient_checkpointing,
            "tensor_parallel_size": self.config.training.tensor_parallel_size,
            "num_train_epochs": self.config.training.num_train_epochs,
            "logging_steps": self.config.training.logging_steps,
            "gradient_accumulation_steps": self.config.training.gradient_accumulation_steps,
        }

        # Train the model.
        current.torch.run(
            torchrun_args={'master_port': '41000'},
            entrypoint="run_clm.py",
            entrypoint_args=entrypoint_args
        )

        # Upload tensor parallel shards.
        model_store = self._get_model_store()
        model_store.upload(
            local_path=checkpoint_dir,
            store_key=os.path.join(
                self.config.model_store.s3_checkpoints_key,
                current.run_id,
                str(current.parallel.node_index),
            ),
        )

        # Upload the neuron compiler cache.
        # Cache contents will be downloaded in future runs to bypass the HF hub cache mechanism and get the training started faster.
        import subprocess
        subprocess.run(["neuron_parallel_compile", "--command", "clear-locks"])
        for subdir in os.listdir(neuron_compiler_cache_dir):
            if subdir in ['lock']:
                continue
            model_store.upload(
                local_path=os.path.join(neuron_compiler_cache_dir, subdir),
                store_key=os.path.join(
                    self.config.model_store.s3_neuron_compiler_cache_key,
                    subdir
                )
            )

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TrainiumLlama2Finetune()
