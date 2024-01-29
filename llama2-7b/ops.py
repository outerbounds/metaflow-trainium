import os
import shutil
from itertools import chain
from typing import Union

from metaflow import S3
from metaflow.metaflow_config import DATATOOLS_S3ROOT

from config import DataStoreConfig, TokenizerStoreConfig, ModelStoreConfig

LLAMA_2_TOKENIZER_DOWNLOAD_INSTRUCTIONS = """To continue, you must download the Llama2 Tokenizer.
Pull the tokenizer from HuggingFace, after being granted access by Meta. 
Detailed instructions for acquiring the tokenizer are available [here](https://huggingface.co/meta-llama/Llama-2-7b). 
When downloading and using the Llama2 Tokenizer and models, you are responsible for adhering to the Meta license.
"""


class BaseStore:

    @classmethod
    def from_path(cls, base_prefix):
        # return cls(os.path.join(DATATOOLS_S3ROOT, base_prefix))
        raise NotImplementedError

    @classmethod
    def from_config(
        cls, config: Union[DataStoreConfig, TokenizerStoreConfig, ModelStoreConfig]
    ):
        return cls(os.path.join(DATATOOLS_S3ROOT, config.s3_prefix))

    def __init__(self, store_root) -> None:
        # store_root is a S3 path to where all files for the store contents will be loaded and saved
        self._store_root = store_root

    @property
    def root(self):
        return self._store_root

    @staticmethod
    def _walk_directory(root):
        path_keys = []
        for path, subdirs, files in os.walk(root):
            for name in files:
                # create a tuple of (key, path)
                path_keys.append(
                    (
                        os.path.relpath(os.path.join(path, name), root),
                        os.path.join(path, name),
                    )
                )
        return path_keys

    def _upload_directory(self, local_path, store_key=""):
        final_path = os.path.join(self._store_root, store_key)
        with S3(s3root=final_path) as s3:
            s3.put_files(self._walk_directory(local_path))

    def already_exists(self, store_key=""):
        final_path = os.path.join(self._store_root, store_key)
        with S3(s3root=final_path) as s3:
            if len(s3.list_paths()) == 0:
                return False
        return True

    def _download_directory(self, download_path, store_key=""):
        """
        Parameters
        ----------
        download_path : str
            Path to the folder where the store contents will be downloaded
        store_key : str
            Key suffixed to the store_root to save the store contents to
        """
        final_path = os.path.join(self._store_root, store_key)
        os.makedirs(download_path, exist_ok=True)
        with S3(s3root=final_path) as s3:
            for s3obj in s3.get_all():
                move_path = os.path.join(download_path, s3obj.key)
                if not os.path.exists(os.path.dirname(move_path)):
                    os.makedirs(os.path.dirname(move_path), exist_ok=True)
                shutil.move(s3obj.path, os.path.join(download_path, s3obj.key))

    def upload(self, local_path, store_key=""):
        """
        Parameters
        ----------
        local_path : str
            Path to the store contents to be saved in cloud object storage.
        store_key : str
            Key suffixed to the store_root to save the store contents to.
        """
        if os.path.isdir(local_path):
            self._upload_directory(local_path, store_key)
        else:
            final_path = os.path.join(self._store_root, store_key)
            with S3(s3root=final_path) as s3:
                s3.put_files([(local_path, local_path)])

    def download(self, download_path, store_key=""):
        """
        Parameters
        ----------
        store_key : str
            Key suffixed to the store_root to download the store contents from
        download_path : str
            Path to the folder where the store contents will be downloaded
        """
        if not self.already_exists(store_key):
            raise ValueError(
                f"Model with key {store_key} does not exist in {self._store_root}"
            )
        self._download_directory(download_path, store_key)


class DataStore(BaseStore):

    @classmethod
    def from_config(cls, config: DataStoreConfig):
        return cls(os.path.join(DATATOOLS_S3ROOT, config.s3_prefix))

    def download_from_huggingface(self, store_config: DataStoreConfig):
        "https://github.com/aws-neuron/neuronx-distributed/blob/main/examples/training/llama2/get_dataset.py"

        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.local_save_path = os.path.abspath(
            os.path.expanduser(store_config.local_path)
        )
        self.tokenizer_path = os.path.abspath(os.path.expanduser(os.getcwd()))

        if not os.path.exists(self.local_save_path):
            os.makedirs(self.local_save_path)

        raw_datasets = load_dataset(
            store_config.hf_dataset_name, store_config.hf_dataset_config_name
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )

        if store_config.block_size > tokenizer.model_max_length:
            print("block_size > tokenizer.model_max_length")
        block_size = min(store_config.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        train_dataset = lm_datasets["train"]
        print(f"Downloaded {len(train_dataset)} pre-training chunks.")

        train_dataset.save_to_disk(self.local_save_path)


class TokenizerStore(BaseStore):

    @property
    def download_instructions(self):
        return LLAMA_2_TOKENIZER_DOWNLOAD_INSTRUCTIONS


class ModelStore(BaseStore):

    def __init__(self, store_root) -> None:
        # store_root is a S3 path to where all files for the store contents will be loaded and saved
        self._store_root = store_root
