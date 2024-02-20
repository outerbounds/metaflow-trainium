from metaflow import FlowSpec, step, resources, environment, current, nvcf
import os
import sys

from config import ConfigBase, EnvironmentConfig, Llama2FinetuneConfig
from ops import DataStore, TokenizerStore, ModelStore
from custom_decorators import pip, enable_decorator
from gpu_profile import gpu_profile

environment_config = EnvironmentConfig()


class Llama2Finetune(FlowSpec, ConfigBase):

    _CORE_CONFIG_CLASS = Llama2FinetuneConfig

    def _get_data_store(self):
        return DataStore.from_config(self.config.data_store)

    def _get_tokenizer_store(self):
        return TokenizerStore.from_config(self.config.tokenizer_store)

    def _get_model_store(self):
        return ModelStore.from_config(self.config.model_store)

    @property
    def config(self) -> Llama2FinetuneConfig:
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
            self.tune_llama2,
            # num_parallel=environment_config.tune_llama2_step.batch_job.n_nodes,
        )

    @gpu_profile(interval=1)
    @pip(packages={**environment_config.tune_llama2_step.packages})
    @environment(vars=environment_config.tune_llama2_step.env_vars)
    @resources(
        gpu=environment_config.tune_llama2_step.batch_job.n_gpu,
        cpu=environment_config.tune_llama2_step.batch_job.n_cpu,
        memory=environment_config.tune_llama2_step.batch_job.memory,
        # image=environment_config.tune_llama2_step.batch_job.image,
    )
    @nvcf(function_id="85457335-38dd-420e-b9a6-2677c552eec2")
    @step
    def tune_llama2(self):

        from omegaconf import OmegaConf
        import json
        import base64

        def make_path(rel_path, make_dir=True, use_tmpfs=False):
            if use_tmpfs:
                path = os.path.join(current.tempdir, rel_path)
            else:
                path = os.path.join(os.getcwd(), rel_path)
            if not os.path.exists(path) and make_dir:
                os.makedirs(path)
            return path

        data_dir = make_path(self.config.data_store.local_path)
        checkpoint_dir = make_path(self.config.model_store.local_checkpoints_path)
        model_path = make_path(self.config.model_store.local_weights_path)

        # Download tokenized data.
        data_store = self._get_data_store()
        data_store.download(download_path=data_dir)

        # convert deepspeed config to json
        model_arch_config = OmegaConf.to_container(self.config.deepspeed)
        ds_config_base64 = base64.urlsafe_b64encode(json.dumps(model_arch_config).encode()).decode()

        # Train the model.
        import subprocess
        cmd = [
            "torchrun",
            "--nproc_per_node", str(environment_config.tune_llama2_step.batch_job.n_gpu),
            "run_clm.py",
            "--model_id", self.config.model_store.hf_model_name,
            "--dataset_path", data_dir,
            "--pretrained_model_cache", "pretrained_model_cache",
            "--bf16", str(self.config.training.bf16),
            "--learning_rate", str(self.config.training.learning_rate),
            "--output_dir", checkpoint_dir,
            "--overwrite_output_dir", str(self.config.training.overwrite_output_dir),
            "--per_device_train_batch_size", str(self.config.training.per_device_train_batch_size),
            "--gradient_checkpointing", str(self.config.training.gradient_checkpointing),
            "--num_train_epochs", str(self.config.training.num_train_epochs),
            "--logging_steps", str(self.config.training.logging_steps),
            "--gradient_accumulation_steps", str(self.config.training.gradient_accumulation_steps),
            "--deepspeed", ds_config_base64
        ]

        # print(cmd)
        subprocess.run(cmd, check=True)
        # with subprocess.Popen(cmd, stdout=subprocess.PIPE,  stderr=subprocess.PIPE) as process:
        #     while process.poll() is None:
        #         stdout = process.stdout.read1()
        #         try:
        #             text = stdout.decode("utf-8")
        #         except UnicodeDecodeError:
        #             text = ""
        #         print(text, end="", flush=True)

        #     if process.returncode != 0:
        #         stderr = process.stderr.read().decode("utf-8")
        #         raise subprocess.CalledProcessError(process.returncode, cmd, stderr)

        # model_store = self._get_model_store()
        # model_store.upload(
        #     local_path=checkpoint_dir,
        #     store_key=os.path.join(
        #         self.config.model_store.s3_checkpoints_key,
        #         current.run_id,
        #         str(current.parallel.node_index),
        #     ),
        # )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    Llama2Finetune()
    # from metaflow import Task; loggen=Task('Llama2Finetune/216402/tune_llama2/1330507').loglines()
