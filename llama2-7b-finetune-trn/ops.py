import os
import shutil
from itertools import chain
from typing import Union
from functools import partial
from random import randint

from metaflow import S3
from metaflow.metaflow_config import DATATOOLS_S3ROOT

from config import DataStoreConfig, TokenizerStoreConfig, ModelStoreConfig

remainder = {"input_ids": [], "attention_mask": [], "token_type_ids": []}


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

    def format_dolly(self, sample):
        instruction = f"### Instruction\n{sample['instruction']}"
        context = (
            f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
        )
        response = f"### Answer\n{sample['response']}"
        # join all the parts together
        prompt = "\n\n".join(
            [i for i in [instruction, context, response] if i is not None]
        )
        return prompt

    # empty list to save remainder from batches to use in next batch
    def pack_dataset(self, dataset, chunk_length=2048):
        """
        https://github.com/huggingface/optimum-neuron/blob/main/notebooks/text-generation/scripts/utils/pack_dataset.py
        """
        print(f"Chunking dataset into chunks of {chunk_length} tokens.")

        def chunk(sample, chunk_length=chunk_length):
            # define global remainder variable to save remainder from batches to use in next batch
            global remainder
            # Concatenate all texts and add remainder from previous batch
            concatenated_examples = {k: list(chain(*sample[k])) for k in sample.keys()}
            concatenated_examples = {
                k: remainder[k] + concatenated_examples[k]
                for k in concatenated_examples.keys()
            }
            # get total number of tokens for batch
            batch_total_length = len(concatenated_examples[list(sample.keys())[0]])

            # get max number of chunks for batch
            if batch_total_length >= chunk_length:
                batch_chunk_length = (batch_total_length // chunk_length) * chunk_length

            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + chunk_length]
                    for i in range(0, batch_chunk_length, chunk_length)
                ]
                for k, t in concatenated_examples.items()
            }
            # add remainder to global variable for next batch
            remainder = {
                k: concatenated_examples[k][batch_chunk_length:]
                for k in concatenated_examples.keys()
            }
            # prepare labels
            result["labels"] = result["input_ids"].copy()
            return result

        # tokenize and chunk dataset
        lm_dataset = dataset.map(
            partial(chunk, chunk_length=chunk_length),
            batched=True,
        )
        print(f"Total number of samples: {len(lm_dataset)}")
        return lm_dataset

    def download_from_huggingface(
        self, store_config: DataStoreConfig, tokenizer_local_path: str
    ):
        "https://github.com/aws-neuron/neuronx-distributed/blob/main/examples/training/llama2/get_dataset.py"

        from transformers import AutoTokenizer
        from datasets import load_dataset
        from random import randrange

        # Load dataset from the hub
        dataset = load_dataset(
            store_config.hf_dataset_name, split=store_config.hf_dataset_split
        )
        # print(f"dataset size: {len(dataset)}")
        # print(dataset[randrange(len(dataset))])
        # dataset size: 15011

        self.local_save_path = os.path.abspath(
            os.path.expanduser(store_config.local_path)
        )
        if not os.path.exists(self.local_save_path):
            os.makedirs(self.local_save_path)

        self.tokenizer_path = os.path.join(
            os.path.abspath(os.path.expanduser(os.getcwd())), tokenizer_local_path
        )
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        # template dataset to add prompt to each sample
        def template_dataset(sample):
            sample["text"] = f"{self.format_dolly(sample)}{tokenizer.eos_token}"
            return sample

        # apply prompt template per sample
        dataset = dataset.map(template_dataset, remove_columns=list(dataset.features))
        # print random sample
        print(dataset[randint(0, len(dataset))]["text"])

        # tokenize dataset
        dataset = dataset.map(
            lambda sample: tokenizer(sample["text"]),
            batched=True,
            remove_columns=list(dataset.features),
        )

        # chunk dataset
        lm_dataset = self.pack_dataset(
            dataset, chunk_length=2048
        )  # We use 2048 as the maximum length for packing
        lm_dataset.save_to_disk(self.local_save_path)


class TokenizerStore(BaseStore):

    def __init__(self, store_root) -> None:
        # store_root is a S3 path to where all files for the store contents will be loaded and saved
        self._store_root = store_root


class ModelStore(BaseStore):

    def __init__(self, store_root) -> None:
        # store_root is a S3 path to where all files for the store contents will be loaded and saved
        self._store_root = store_root
