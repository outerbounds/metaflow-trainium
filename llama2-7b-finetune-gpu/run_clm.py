from dataclasses import dataclass, field

from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


def training_function(script_args, training_args):
    dataset = load_from_disk(script_args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        cache_dir=script_args.pretrained_model_cache,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model()


@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
        default="philschmid/Llama-2-7b-hf",
    )
    dataset_path: str = field(
        metadata={"help": "Path to the preprocessed and tokenized dataset."},
        default="data/databricks-dolly-15k",
    )
    pretrained_model_cache: str = field(
        metadata={"help": "Path to the preprocessed and tokenized dataset."},
        default="/metaflow_temp/pretrained_model_cache",
    )


def main():
    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()
