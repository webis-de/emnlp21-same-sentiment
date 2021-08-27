# https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py

import json
import logging
import os
import sys
from pprint import pformat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )


# ---------------------------------------------------------------------------

import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
)

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.optimization import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from transformers.trainer_utils import is_main_process

from datasets import SamenessDataset
from datasets import SamenessDataTrainingArguments as DataTrainingArguments
from processors import sameness_output_modes, sameness_tasks_num_labels
from metrics import sameness_compute_metrics


# ---------------------------------------------------------------------------


@dataclass
class MyTrainingArguments(TrainingArguments):
    do_test: bool = field(
        default=False, metadata={"help": "Whether to run evaluation on the test set."}
    )


class MyTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)

        # self.lr_scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        # )
        self.lr_scheduler = get_constant_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_steps
        )


# ---------------------------------------------------------------------------


def store_run_args(base_path, extra=None):
    if extra is None:
        extra = ""
    else:
        extra = f".{extra}"

    output_argv_file = os.path.join(base_path, f"argv{extra}.txt")

    with open(output_argv_file, "w") as writer:
        for item in sys.argv:
            if not item:
                continue
            writer.write(f"{item}\n")


# ---------------------------------------------------------------------------


def run(json_config_file=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments)
    )

    if json_config_file is not None:
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=json_config_file
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # -----------------------------------------------------------------------

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # dataclasses.asdict
    # TODO: serialize current run args

    # -----------------------------------------------------------------------

    # Setup logging
    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if is_main_process(training_args.local_rank)
        else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Dataset parameters {data_args}")
    logger.info(f"Model parameters {model_args}")

    # -----------------------------------------------------------------------

    # Set seed
    set_seed(training_args.seed)

    # -----------------------------------------------------------------------

    try:
        num_labels = sameness_tasks_num_labels[data_args.task_name]
        output_mode = sameness_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # -----------------------------------------------------------------------

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # -----------------------------------------------------------------------

    # Get datasets
    train_dataset = (
        SamenessDataset(
            data_args, tokenizer=tokenizer, mode="train", cache_dir=model_args.cache_dir
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        SamenessDataset(
            data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        SamenessDataset(
            data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir
        )
        if training_args.do_test
        else None
    )
    pred_dataset = (
        SamenessDataset(
            data_args, tokenizer=tokenizer, mode="pred", cache_dir=model_args.cache_dir
        )
        if training_args.do_predict
        else None
    )

    # -----------------------------------------------------------------------

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return sameness_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    # -----------------------------------------------------------------------

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )

    # -----------------------------------------------------------------------

    # write out argv
    if training_args.do_train:
        if trainer.is_world_process_zero():
            store_run_args(training_args.output_dir, extra="train")
    elif training_args.do_eval:
        if trainer.is_world_process_zero():
            store_run_args(training_args.output_dir, extra="eval")
    elif training_args.do_test:
        if trainer.is_world_process_zero():
            store_run_args(training_args.output_dir, extra="test")

    # -----------------------------------------------------------------------

    # Training
    if training_args.do_train:
        # EK: for more convenience, store tokenizer so we can easily copy it
        # into the checkpoint folders for evaluating aborted runs
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)
        # run training
        trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # -----------------------------------------------------------------------

    def run_eval(eval_dataset, mode: str = "dev", eval_filename_prefix: str = "eval"):
        logger.info(f"*** Evaluate '{mode}' ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                SamenessDataset(
                    mnli_mm_data_args,
                    tokenizer=tokenizer,
                    mode=mode,
                    cache_dir=model_args.cache_dir,
                )
            )

        eval_results = dict()
        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(
                eval_dataset.args.task_name
            )
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir,
                f"{eval_filename_prefix}_results_{eval_dataset.args.task_name}.txt",
            )
            output_eval_file_json = os.path.join(
                training_args.output_dir,
                f"{eval_filename_prefix}_results_{eval_dataset.args.task_name}.json",
            )
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(
                        f"***** Eval '{mode}' results {eval_dataset.args.task_name} *****"
                    )
                    for key, value in eval_result.items():
                        logger.info(f"  {key} = {pformat(value)}")
                        writer.write(f"{key} = {value}\n")
                with open(output_eval_file_json, "w") as writer:
                    json.dump(eval_result, writer, indent=2)

            eval_results.update(eval_result)

        return eval_results

    # -----------------------------------------------------------------------

    # Validation evaluation
    eval_results = {}
    if training_args.do_eval:
        eval_result = run_eval(eval_dataset, "dev", "eval")

    # Test evaluation
    test_results = {}
    if training_args.do_test:
        test_result = run_eval(test_dataset, "test", "test")

    # -----------------------------------------------------------------------

    if training_args.do_predict:
        logging.info("*** Test prediction ***")
        pred_datasets = [pred_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            pred_datasets.append(
                SamenessDataset(
                    mnli_mm_data_args,
                    tokenizer=tokenizer,
                    mode="pred",
                    cache_dir=model_args.cache_dir,
                )
            )

        for pred_dataset in pred_datasets:
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # pred_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=pred_dataset).predictions
            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)
            elif output_mode == "regression":
                predictions = np.squeeze(predictions)

            output_pred_file = os.path.join(
                training_args.output_dir,
                f"pred_results_{pred_dataset.args.task_name}.txt",
            )
            if trainer.is_world_process_zero():
                with open(output_pred_file, "w") as writer:
                    logger.info(
                        "***** Prediction results {} *****".format(
                            pred_dataset.args.task_name
                        )
                    )
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = pred_dataset.get_labels()[item]
                            writer.write(f"{index}\t{item}\n")

    # -----------------------------------------------------------------------

    return eval_results, test_results


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    json_config_file = "01_base_transformer_config.json"
    json_config_file = None
    run(json_config_file=json_config_file)


# ---------------------------------------------------------------------------
