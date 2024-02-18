import os
import shutil
from typing import Union
from random import randint

from metaflow import S3
from metaflow.metaflow_config import DATATOOLS_S3ROOT

from config import DataStoreConfig, TokenizerStoreConfig, ModelStoreConfig


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

    def download_from_huggingface(
        self, store_config: DataStoreConfig, tokenizer_local_path: str
    ):
        "https://huggingface.co/docs/optimum-neuron/en/tutorials/fine_tune_bert"

        from transformers import AutoTokenizer
        from datasets import load_dataset
        from random import randrange

        # Load dataset from the hub
        dataset = load_dataset(
            store_config.hf_dataset_name,  # split=store_config.hf_dataset_split
        )

        self.local_save_path = os.path.abspath(
            os.path.expanduser(store_config.local_path)
        )
        if not os.path.exists(self.local_save_path):
            os.makedirs(self.local_save_path)

        self.tokenizer_path = os.path.join(
            os.path.abspath(os.path.expanduser(os.getcwd())), tokenizer_local_path
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        # Tokenize dataset
        dataset = dataset.rename_column("label", "labels")  # to match Trainer
        tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        tokenized_dataset = tokenized_dataset.with_format("torch")

        # save dataset to disk
        tokenized_dataset["train"].save_to_disk(
            os.path.join(self.local_save_path, "train")
        )
        tokenized_dataset["test"].save_to_disk(
            os.path.join(self.local_save_path, "eval")
        )


class TokenizerStore(BaseStore):

    def __init__(self, store_root) -> None:
        # store_root is a S3 path to where all files for the store contents will be loaded and saved
        self._store_root = store_root


class ModelStore(BaseStore):

    def __init__(self, store_root) -> None:
        # store_root is a S3 path to where all files for the store contents will be loaded and saved
        self._store_root = store_root
