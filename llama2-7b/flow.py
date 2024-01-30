import os
import sys
import json
from multiprocessing import Pool, TimeoutError
from tempfile import NamedTemporaryFile

from metaflow import FlowSpec, step, batch, torchrun, current, S3, environment, Parameter, card
from metaflow.plugins.parallel_decorator import UBF_CONTROL

from config import ConfigBase, EnvironmentConfig, TrainiumLlama2PretrainConfig
from ops import DataStore, TokenizerStore, ModelStore
from custom_decorators import pip, enable_decorator

environment_config = EnvironmentConfig()


class TrainiumLlama2Pretrain(FlowSpec, ConfigBase):

    # Use the checkpoint parameter in the Metaflow run command to resume training from a checkpoint.
    # The value must be a valid checkpoint key in the model_store, currently set up to be indexed my the Metaflow run_id.
    remote_checkpoint_id = Parameter(name="checkpoint", default=None, type=str, help="checkpoint to resume training from")

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

    @pip(packages={**environment_config.dataset_cache_step.packages})
    @enable_decorator(batch, environment_config.dataset_cache_step.batch_enabled)
    @step
    def cache_dataset(self):
        store = self._get_data_store()
        if not store.already_exists():
            store.download_from_huggingface(self.config.data_store)
            store.upload(self.config.data_store.local_path)
        self.next(self.train_llama2, num_parallel=environment_config.train_llama2_step.batch_job.n_nodes)

    @environment(vars=environment_config.train_llama2_step.env_vars)
    @batch(
        # NOTE: Trainium and inferentia modify batch job submitted in the same way. 
        # So, we can use the same `inferentia` arg for both for now.
        # TODO: change Metaflow to say neuron here for either case.
        inferentia=environment_config.train_llama2_step.batch_job.n_trainium_devices,
        efa=environment_config.train_llama2_step.batch_job.n_efa_interfaces,
        cpu=environment_config.train_llama2_step.batch_job.n_cpu,
        memory=environment_config.train_llama2_step.batch_job.memory,
        image=environment_config.train_llama2_step.batch_job.image,
        queue=environment_config.train_llama2_step.batch_job.job_queue,
    )
    @torchrun
    @step
    def train_llama2(self):
        from omegaconf import OmegaConf
        import json

        # Download tokenized data.
        data_store = self._get_data_store()
        data_store.download(download_path=self.config.data_store.local_path)

        # Download checkpoint from model_store.
        model_store = self._get_model_store()
        resume_checkpoint_arg = {}
        if self.remote_checkpoint_id is not None:
            model_store.download(
                download_path=self.config.model_store.local_checkpoints_path,
                store_key=os.path.join(
                    self.config.model_store.s3_checkpoints_key, 
                    current.parallel.node_index,
                    self.remote_checkpoint_id
                )
            )
            resume_checkpoint_arg['resume_ckpt'] = None # NOTE: None tells current.torch.run to use store_true style command line arg

        # Create model_store.local_weights_path directory relative to working directory.
        # This is where model weights and config go, NOT the checkpoints.
        def make_path(rel_path, make_dir=True):
            path = os.path.join(os.getcwd(), rel_path)
            if not os.path.exists(path) and make_dir:
                os.makedirs(path)
            return path
        data_dir = make_path(self.config.data_store.local_path)
        model_path = make_path(self.config.model_store.local_weights_path)
        checkpoint_dir = make_path(self.config.model_store.local_checkpoints_path)
        metrics_file = make_path(self.config.training.metrics_file, make_dir=False)

        # Write config.json used by transformers model. If desired, you could alternatively package a hard-coded config.json in the Docker image.
        model_arch_config = OmegaConf.to_container(self.config.model_architecture)
        with open(os.path.join(model_path, "config.json"), "w") as f:
            json.dump(model_arch_config, f)

        # Configure entrypoint args.
        world_size = (
            environment_config.train_llama2_step.batch_job.n_nodes 
            * environment_config.train_llama2_step.batch_job.n_trainium_devices 
            * 2
        )
        data_parallelism_degree = world_size / self.config.training.tensor_parallelism_degree
        # accumulation_steps = int(
        #     self.config.training.global_batch_size
        #     / self.config.training.micro_batch_size
        #     / data_parallelism_degree
        # )
        accumulation_steps = 1 # TESTING ~ only enter checkpointing logic if training_ustep % grad_accum_usteps == 0
        entrypoint_args = {
            "model_path": model_path, # TODO: rel path did not to work, so using abs path.  
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
            # "force_checkpoint": None, # NOTE: Checkpoints every epoch no matter what, included for testing.
        }
        if self.config.training.use_mix_precision:
            entrypoint_args["use_mix_precision"] = None
        if self.config.training.use_zero_1:
            entrypoint_args["use_zero_1"] = None

        # Run llama2 pretraining. @torchrun exposes the current.torch.run action, which constructs the distributed training pieces of the command.
        current.torch.run(
            entrypoint="tp_zero1_llama2_7b_hf_pretrain.py",
            entrypoint_args=entrypoint_args,
            master_port="41000" # NOTE: 41000 is hardcoded in reserved ports in the Dockerfile.
        )

        # Upload checkpoint artifacts for use in continued pre-training or next stage (e.g., instruction tuning).
        try:
            model_store.upload(
                local_path=checkpoint_dir,
                store_key=os.path.join(
                    self.config.model_store.s3_checkpoints_key, 
                    current.parallel.node_index,
                    current.run_id
                )
            )
        except:
            import time
            print("Failed checkpoint upload sleeping 2 hrs")
            time.sleep(2*3600)

        # Push TensorBoard logs to S3.
        experiment_logs = os.path.join(os.getcwd(), "outputs")
        try:
            model_store.upload(experiment_logs, store_key=current.run_id)
            model_store.upload(
                local_path=experiment_logs,
                store_key=os.path.join(
                    self.config.model_store.s3_experiments_key, 
                    current.parallel.node_index,
                    current.run_id
                )
            )
        except:
            print("Failed experiment logs upload sleeping 2 hrs")
            time.sleep(2*3600)

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        print("""Flow completed successfully.""")


if __name__ == "__main__":
    TrainiumLlama2Pretrain()
