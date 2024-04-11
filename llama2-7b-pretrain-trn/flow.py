import os
import sys
import json

from metaflow import (
    FlowSpec,
    step,
    batch,
    torchrun,
    current,
    S3,
    environment,
    Parameter,
    card,
)
from metaflow.plugins.parallel_decorator import UBF_CONTROL

from config import ConfigBase, EnvironmentConfig, TrainiumLlama2PretrainConfig
from ops import DataStore, TokenizerStore, ModelStore
from custom_decorators import pip, enable_decorator
from neuron_monitor import neuron_monitor

environment_config = EnvironmentConfig()


class TrainiumLlama2Pretrain(FlowSpec, ConfigBase):

    # Use the checkpoint parameter in the Metaflow run command to resume training from a checkpoint.
    # The value must be a valid checkpoint key in the model_store, currently set up to be indexed my the Metaflow run_id.
    remote_checkpoint_id = Parameter(
        name="checkpoint",
        default=None,
        type=str,
        help="checkpoint to resume training from",
    )

    _CORE_CONFIG_CLASS = TrainiumLlama2PretrainConfig

    def _get_data_store(self):
        return DataStore.from_config(self.config.data_store)

    def _get_tokenizer_store(self):
        return TokenizerStore.from_config(self.config.tokenizer_store)

    def _get_model_store(self):
        return ModelStore.from_config(self.config.model_store)

    @property
    def config(self) -> TrainiumLlama2PretrainConfig:
        return self._get_config()

    @pip(packages={"omegaconf": "2.3.0"})
    @step
    def start(self):
        store = self._get_tokenizer_store()
        if not store.already_exists():
            if os.path.exists(self.config.tokenizer_store.local_path):
                store.upload(self.config.tokenizer_store.local_path)
            else:
                sys.exit(store.download_instructions)
        self.next(self.cache_dataset)

    def _write_config_as_json(self, relpath=""):
        from omegaconf import OmegaConf
        import json

        model_arch_config = OmegaConf.to_container(self.config.model_architecture)
        with open(os.path.join(relpath, "config.json"), "w") as f:
            json.dump(model_arch_config, f)

    @pip(packages={**environment_config.dataset_cache_step.packages})
    @enable_decorator(batch, environment_config.dataset_cache_step.batch_enabled)
    @step
    def cache_dataset(self):
        data_store = self._get_data_store()
        if not data_store.already_exists():
            self._write_config_as_json()
            if not os.path.exists(self.config.tokenizer_store.local_path):
                tokenizer_store = self._get_tokenizer_store()
                tokenizer_store.download(
                    download_path=self.config.tokenizer_store.local_path
                )
            data_store.download_from_huggingface(self.config.data_store)
            data_store.upload(self.config.data_store.local_path)
        self.next(
            self.train_llama2,
            num_parallel=environment_config.train_llama2_step.batch_job.n_nodes,
        )

    @environment(vars=environment_config.train_llama2_step.env_vars)
    @batch(
        trainium=environment_config.train_llama2_step.batch_job.n_trainium_devices,
        efa=environment_config.train_llama2_step.batch_job.n_efa_interfaces,
        cpu=environment_config.train_llama2_step.batch_job.n_cpu,
        memory=environment_config.train_llama2_step.batch_job.memory,
        image=environment_config.train_llama2_step.batch_job.image,
        queue=environment_config.train_llama2_step.batch_job.job_queue,
        use_tmpfs=True,  # size is 1/2 of `memory` by default.
    )
    @neuron_monitor(interval=1)
    @torchrun
    @step
    def train_llama2(self):

        # Create model_store.local_weights_path directory relative to working directory.
        # This is where model weights and config go, NOT the checkpoints.
        def make_path(rel_path, make_dir=True, use_tmpfs=False):
            if use_tmpfs:
                path = os.path.join(current.tempdir, rel_path)
            else:
                path = os.path.join(os.getcwd(), rel_path)
            if not os.path.exists(path) and make_dir:
                os.makedirs(path)
            return path

        data_dir = make_path(self.config.data_store.local_path)
        model_path = make_path(self.config.model_store.local_weights_path)
        checkpoint_dir = make_path(
            self.config.model_store.local_checkpoints_path, use_tmpfs=True
        )
        metrics_file = make_path(self.config.training.metrics_file, make_dir=False)

        # Download tokenized data.
        data_store = self._get_data_store()
        data_store.download(download_path=data_dir)

        # Download checkpoint from model_store.
        model_store = self._get_model_store()
        resume_checkpoint_arg = {}
        if self.remote_checkpoint_id is not None:
            # NOTE: This downloads all checkpoints from the specified run_id.
            model_store.download(
                download_path=checkpoint_dir,
                store_key=os.path.join(
                    self.config.model_store.s3_checkpoints_key,
                    self.remote_checkpoint_id,
                    current.parallel.node_index,
                ),
            )
            resume_checkpoint_arg[
                "resume_ckpt"
            ] = None  # NOTE: value None tells current.torch.run to use store_true style command line arg

        # Write config.json used by transformers model. If desired, you could alternatively package a hard-coded config.json in the Docker image.
        self._write_config_as_json(relpath=model_path)

        # Configure entrypoint args.
        world_size = (
            environment_config.train_llama2_step.batch_job.n_nodes
            * environment_config.train_llama2_step.batch_job.n_trainium_devices
            * 2  # cores per device
        )
        data_parallelism_degree = (
            world_size / self.config.training.tensor_parallelism_degree
        )
        accumulation_steps = int(
            self.config.training.global_batch_size
            / self.config.training.micro_batch_size
            / data_parallelism_degree
        )
        entrypoint_args = {
            "model_path": model_path,
            "data_dir": data_dir,
            "tensor_parallel_size": self.config.training.tensor_parallelism_degree,
            "batch_size": self.config.training.micro_batch_size,
            "steps_this_run": self.config.training.steps_this_run,
            "max_steps": self.config.training.total_steps,
            "warmup_steps": self.config.training.warmup_steps,
            "lr": self.config.training.learning_rate,
            "grad_accum_usteps": accumulation_steps,
            "seq_len": self.config.training.sequence_length,
            "sequence_parallel_enabled": None,
            "selective_checkpoint_enabled": None,
            "logging_interval": self.config.training.logging_interval,
            **resume_checkpoint_arg,
            "checkpoint_dir": checkpoint_dir,
            "metrics_file": metrics_file,
            # "force_checkpoint": None, # NOTE: Checkpoints every step no matter what, included for testing.
        }
        if self.config.training.use_mix_precision:
            entrypoint_args["use_mix_precision"] = None
        if self.config.training.use_zero_1:
            entrypoint_args["use_zero_1"] = None

        # Run llama2 pretraining. @torchrun exposes the current.torch.run action, which constructs the distributed training pieces of the command.
        current.torch.run(
            torchrun_args={
                "master_port": "41000",  # NOTE: 41000 is hardcoded in reserved ports in the Dockerfile.
            },
            entrypoint="tp_zero1_llama2_7b_hf_pretrain.py",
            entrypoint_args=entrypoint_args,
            # master_port="41000",  # NOTE: 41000 is hardcoded in reserved ports in the Dockerfile.
        )

        # Push checkpoints for use in continued pre-training or next stage (e.g., instruction tuning).
        model_store.upload(
            local_path=checkpoint_dir,
            store_key=os.path.join(
                self.config.model_store.s3_checkpoints_key,
                current.run_id,
                str(current.parallel.node_index),
            ),
        )

        if current.parallel.node_index == 0:

            # Push TensorBoard logs to S3.
            experiment_logs = os.path.join(os.getcwd(), "output")
            model_store.upload(
                local_path=experiment_logs,
                store_key=os.path.join(
                    self.config.model_store.s3_experiments_key, current.run_id
                ),
            )

            # Store metrics file.
            import json

            self.metrics_json = json.load(open(metrics_file))

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print("""Flow completed successfully.""")


if __name__ == "__main__":
    TrainiumLlama2Pretrain()
