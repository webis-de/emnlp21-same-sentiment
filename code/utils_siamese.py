# https://github.com/huggingface/transformers/blob/9f72e8f4e1e767c5f608dd135199e592255b8a69/src/transformers/data/processors/utils.py

import csv
import dataclasses
import json
import logging
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``tf``.
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------


def load_predictions(folder, mode, dtype=int):
    if isinstance(mode, Split):
        mode = mode.value
    with open(os.path.join(folder, f"{mode}_results.txt"), "r") as fp:
        fp.readline()
        preds = [dtype(line.strip().rsplit("\t", 1)[-1]) for line in fp]
    return np.array(preds)


def load_truths(folder, mode):
    if isinstance(mode, Split):
        mode = mode.value
    with open(os.path.join(folder, f"{mode}.tsv"), "r") as fp:
        truths = [int(line.strip().rsplit("\t", 1)[-1]) for line in fp]
    return np.array(truths)


def compute_metrics(y_true, y_pred, precision=8):
    # https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-and-f-measures

    metrics = {
        "class_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["not-same", "same"],
            digits=precision,
            output_dict=True,
        ),
        "confusion": {
            ln: v
            for ln, v in zip(
                ["tn", "fp", "fn", "tp"], confusion_matrix(y_true, y_pred).ravel()
            )
        },
        "accuracy": accuracy_score(y_true, y_pred),
        **{
            ln: v
            for ln, v in zip(
                ["precision", "recall", "fscore", "support"],
                precision_recall_fscore_support(y_true, y_pred, average="binary"),
            )
            if ln != "support"
        },
    }

    return metrics


# ---------------------------------------------------------------------------


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


# ---------------------------------------------------------------------------


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    pred = "pred"


# ---------------------------------------------------------------------------


class SameSentimentDataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """
        Gets an example from a dict with tensorflow tensors.
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the train set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the dev set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev"
        )

    def get_test_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the test set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test"
        )

    def get_pred_examples(self, data_dir):
        """Get prediction examples (no label in dataset)."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "pred.tsv")), "pred"
        )

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["0", "1"]

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

    def tfds_map(self, example):
        """
        Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. This method converts
        examples to the correct format.
        """
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


# ---------------------------------------------------------------------------


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    # https://stackoverflow.com/a/61903895/9360161

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


# ---------------------------------------------------------------------------
