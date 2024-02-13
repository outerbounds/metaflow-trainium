from metaflow import FlowSpec, step, batch, torchrun, environment, current
import os
import sys

from config import ConfigBase, EnvironmentConfig, TrainiumLlama2FinetuneConfig
from ops import DataStore, TokenizerStore, ModelStore
from custom_decorators import pip, enable_decorator
# from neuron_monitor import neuron_monitor

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
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_store.hf_model_name)
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
            data_store.download_from_huggingface(self.config.data_store, self.config.tokenizer_store.local_path)
            data_store.upload(self.config.data_store.local_path)
        self.next(self.tune_llama2, num_parallel=environment_config.train_llama2_step.batch_job.n_nodes)

    @pip(packages={**environment_config.train_llama2_step.packages})
    @environment(vars=environment_config.train_llama2_step.env_vars)
    @batch(
        inferentia=environment_config.train_llama2_step.batch_job.n_trainium_devices,
        efa=environment_config.train_llama2_step.batch_job.n_efa_interfaces,
        cpu=environment_config.train_llama2_step.batch_job.n_cpu,
        memory=environment_config.train_llama2_step.batch_job.memory,
        image=environment_config.train_llama2_step.batch_job.image,
        queue=environment_config.train_llama2_step.batch_job.job_queue,
        use_tmpfs=True, # size is 1/2 of `memory` by default.
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

        # Download tokenized data.
        data_store = self._get_data_store()
        data_store.download(download_path=data_dir)

        self.compile = False
        if self.compile: 
            pass 

        # MALLOC_ARENA_MAX=64 torchrun --nproc_per_node=32 scripts/run_clm.py \
        #     --model_id {model_id} \
        #     --dataset_path {dataset_path} \
        #     --bf16 True \
        #     --learning_rate 5e-5 \
        #     --output_dir dolly_llama \
        #     --overwrite_output_dir True \
        #     --skip_cache_push True \
        #     --per_device_train_batch_size 1 \
        #     --gradient_checkpointing True \
        #     --tensor_parallel_size 8 \
        #     --num_train_epochs 3 \
        #     --logging_steps 10 \
        #     --gradient_accumulation_steps 16

        # Train the model.
        # current.torch.run(
        #     entrypoint="run_clm.py",
        #     entrypoint_args={
        #         "model_id": self.config.model_store.hf_model_name,
        #         "dataset_path": data_dir
        #     },
        #     master_port="41000",
        # )

        import time
        print("\n\n\nSleeping for 2 hours")
        time.sleep(3600 * 2)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    TrainiumLlama2Finetune()