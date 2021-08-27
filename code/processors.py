# https://github.com/huggingface/transformers/blob/9e9a1fb8c75e2ef00fea9c4c0dc511fc0178081c/src/transformers/data/processors/glue.py

"""Sameness processors and helpers """

import os
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from transformers.data.processors.utils import (
    DataProcessor,
    InputExample,
    InputFeatures,
)

# from transformers.data.processors.glue import OutputMode, glue_convert_examples_to_features


# ---------------------------------------------------------------------------


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Converters


def sameness_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    padding: str = "max_length",
    truncation: bool = True,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: Sameness task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    return _sameness_convert_examples_to_features(
        examples,
        tokenizer,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        task=task,
        label_list=label_list,
        output_mode=output_mode,
    )


def _sameness_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    padding: str = "max_length",
    truncation: bool = True,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = sameness_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = sameness_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        elif output_mode == "regression5":
            return float(example.label) * 4.0 + 1.0
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding=padding,
        truncation=truncation,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


# ---------------------------------------------------------------------------


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


# ---------------------------------------------------------------------------
# Processors


class SamenessProcessorBase(DataProcessor):
    """Base Processor for the Sameness data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """Get test examples for holdout evaluation."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_pred_examples(self, data_dir):
        """Get prediction examples (no label in dataset)."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "pred.tsv")), "pred"
        )

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # TODO: has headers?
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = None if set_type == "pred" else line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class SamenessProcessorClassification(SamenessProcessorBase):
    """Processor for the Sameness data set."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


class SamenessProcessorRegression(SamenessProcessorBase):
    """Processor for the Sameness data set."""

    def get_labels(self):
        """See base class."""
        return [None]


# ---------------------------------------------------------------------------


class SingleSentenceProcessorBase(DataProcessor):
    """Processor for the single sentence sentiment? prediction data set (like CoLA for GLUE)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """Get test examples for holdout evaluation."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_pred_examples(self, data_dir):
        """Get prediction examples (no label in dataset)."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "pred.tsv")), "pred"
        )

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[2]
            label = None if set_type == "pred" else line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples


class SingleSentenceProcessorClassification(SingleSentenceProcessorBase):
    """Processor for the single sentence score data set."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


class SingleSentenceProcessorClassification5(SingleSentenceProcessorBase):
    """Processor for the single sentence score data set."""

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4", "5"]


class SingleSentenceProcessorRegression(SingleSentenceProcessorBase):
    """Processor for the single sentence score data set."""

    def get_labels(self):
        """See base class."""
        return [None]


# ---------------------------------------------------------------------------
# Lookups


sameness_tasks_num_labels = {
    "same-b": 2,
    "same-r": 1,
    "sent-b": 2,
    "sent-r": 1,
    # "sent-r5": 1,#5,
    "sent-5": 5,
}


sameness_processors = {
    "same-b": SamenessProcessorClassification,
    "same-r": SamenessProcessorRegression,
    "sent-b": SingleSentenceProcessorClassification,
    "sent-r": SingleSentenceProcessorRegression,
    # "sent-r5": SingleSentenceProcessorRegression,
    "sent-5": SingleSentenceProcessorClassification5,
}


sameness_output_modes = {
    "same-b": "classification",
    "same-r": "regression",
    "sent-b": "classification",
    "sent-r": "regression",
    # "sent-r5": "regression5",
    "sent-5": "classification",
}


# ---------------------------------------------------------------------------
