# https://github.com/huggingface/transformers/blob/9e9a1fb8c75e2ef00fea9c4c0dc511fc0178081c/src/transformers/data/datasets/glue.py

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

# from transformers.data.processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from transformers.data.processors.utils import InputFeatures

from processors import (
    sameness_convert_examples_to_features,
    sameness_output_modes,
    sameness_processors,
)


# ---------------------------------------------------------------------------


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Arguments


@dataclass
class SamenessDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(sameness_processors.keys())
        }
    )
    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."
        }
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    # train_file: Optional[str] = field(
    #    default=None, metadata={"help": "A csv or a json file containing the training data."}
    # )
    # validation_file: Optional[str] = field(
    #    default=None, metadata={"help": "A csv or a json file containing the validation data."}
    # )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


# ---------------------------------------------------------------------------


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    pred = "pred"


# ---------------------------------------------------------------------------
# Dataset


class SamenessDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: SamenessDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: SamenessDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = sameness_processors[args.task_name]()
        self.output_mode = sameness_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__.__name__ in (
            "RobertaTokenizer",
            "RobertaTokenizerFast",
            "XLMRobertaTokenizer",
            "BartTokenizer",
            "BartTokenizerFast",
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]",
                    time.time() - start,
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                elif mode == Split.pred:
                    examples = self.processor.get_pred_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                if limit_length is not None:
                    examples = examples[:limit_length]

                # Padding strategy
                if args.pad_to_max_length:
                    padding = "max_length"
                    max_length = args.max_seq_length
                else:
                    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
                    padding = False
                    max_length = None

                self.features = sameness_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=max_length,
                    padding=padding,
                    truncation=True,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]",
                    cached_features_file,
                    time.time() - start,
                )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


# ---------------------------------------------------------------------------
