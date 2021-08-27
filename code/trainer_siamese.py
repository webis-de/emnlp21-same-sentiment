# ---

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

# ---

import json
import logging
import os
import gc
import pickle
import re
import string
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Input,
    Dense,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# from tqdm.keras import TqdmCallback

from hf_argparser import HfArgumentParser

from utils_siamese import (
    SameSentimentDataProcessor,
    Split,
    set_seed,
    NumpyEncoder,
    compute_metrics,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


@dataclass
class MyTrainingArguments:
    output_dir: str = field(
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."
        }
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    fn_vectors: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to glove embeddings. Will infer from `data_dir` and `embedding_dim`."
        },
    )

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model checkpoint"}
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, metadata={"help": "Whether to run eval on the dev set."}
    )
    do_test: bool = field(
        default=False, metadata={"help": "Whether to run evaluation on the test set."}
    )
    do_predict: bool = field(
        default=False, metadata={"help": "Whether to run predictions on the test set."}
    )

    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )

    # 200
    num_epochs: int = field(
        default=20, metadata={"help": "Total number of training epochs to perform."}
    )
    # 512  # 1024  # 64
    per_device_train_batch_size: int = field(
        default=512, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=512,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )

    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    discard_too_long_samples: bool = field(
        default=False,
        metadata={
            "help": "Whether to discard all samples longer than `max_seq_length`. "
            "If False, will truncate and pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    embedding_dim: int = field(
        default=50,
        metadata={"help": "Dimension of glove embeddings (50, 100, 200, 300)."},
    )
    rate_drop_lstm: float = field(
        default=0.17, metadata={"help": "The drop rate for LSTM layers."}
    )
    rate_drop_dense: float = field(
        default=0.25, metadata={"help": "The drop rate for Dense layers."}
    )
    # 15
    number_lstm: int = field(default=50, metadata={"help": "The number of LSTM cells."})
    number_dense_units: int = field(
        default=50, metadata={"help": "The number of dense units."}
    )
    activation_function: str = field(
        default="relu", metadata={"help": "The activation function."}
    )

    print_model: bool = field(default=False, metadata={"help": "Print model summary."})

    def __post_init__(self):
        assert self.embedding_dim in (50, 100, 200, 300)
        if not self.fn_vectors:
            self.fn_vectors = os.path.join(
                self.data_dir, f"glove.6B.{self.embedding_dim}d.txt"
            )
        if not self.cache_dir:
            self.cache_dir = self.data_dir


# ---------------------------------------------------------------------------


def load_tsv_data(
    data_dir: str, processor: SameSentimentDataProcessor, mode: Optional[Split] = None
):
    if mode == Split.dev:
        examples = processor.get_dev_examples(data_dir)
    elif mode == Split.test:
        examples = processor.get_test_examples(data_dir)
    elif mode == Split.pred:
        examples = processor.get_pred_examples(data_dir)
    else:
        examples = processor.get_train_examples(data_dir)

    guid, text_a, text_b, label = list(), list(), list(), list()
    for example in examples:
        guid.append(example.guid)
        text_a.append(example.text_a)
        text_b.append(example.text_b)
        label.append(example.label)

    guid = np.array(guid)
    text_a = np.array(text_a)
    text_b = np.array(text_b)
    label = np.array(label)

    if mode == Split.pred:
        label = None

    return guid, text_a, text_b, label


# ---------------------------------------------------------------------------


def clean_text(lines):
    """Clean text by removing unnecessary characters and altering the format of words."""
    re_print = re.compile("[^%s]" % re.escape(string.printable))
    cleaned = list()
    for text in lines:
        text = text.lower()

        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        # remove punctuation
        text = re.sub(r"[$-()\"#/@;:<>{}`+=~|.!?,'*-]", "", text)

        text = text.split()
        text = [re_print.sub("", w) for w in text]

        cleaned.append(" ".join(text))

    cleaned = np.array(cleaned)

    return cleaned


# ---------------------------------------------------------------------------


def read_glove_vectors(path: str):
    """
    read Glove Vector Embeddings
    """

    with open(path, encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            cur_word = line[0]
            words.add(cur_word)
            word_to_vec_map[cur_word] = np.array(line[1:], dtype=np.float64)

    i = 1
    words_to_index = {}
    index_to_words = {}
    for w in sorted(words):
        words_to_index[w] = i
        index_to_words[i] = w
        i = i + 1

    return words_to_index, index_to_words, word_to_vec_map


def update_special_tokens(vocab_to_int, int_to_vocab, word_to_vec_map):
    """Special Tokens"""
    for code in ["<PAD>", "<EOS>", "<UNK>", "<GO>"]:
        vocab_to_int[code] = len(vocab_to_int) + 1
        int_to_vocab[len(int_to_vocab) + 1] = code
        word_to_vec_map[code] = np.random.random(50)

    return vocab_to_int, int_to_vocab, word_to_vec_map


# ---------------------------------------------------------------------------


def _trim_too_long(length: int, sentences):
    for idx in range(len(sentences)):
        tokens = sentences[idx].strip().split()
        tokens = tokens[:length]
        sentences[idx] = " ".join(tokens)
    return sentences


def preprocess(fn_data, vocab_to_int, max_len, mode=None, filter_too_long=False):
    processor = SameSentimentDataProcessor()

    guid, sent1, sent2, labels = load_tsv_data(fn_data, processor, mode)

    # --------------------------------

    # clean text, remove contractions, tokenize
    sent1 = clean_text(sent1)
    sent2 = clean_text(sent2)

    # --------------------------------

    if filter_too_long:
        # remove everything to long
        def get_short_indices(length, sentences):
            """
            Filter out sequences with length "length"
            """
            idx = np.zeros((len(sentences)), dtype=bool)
            for num, sent in enumerate(sentences):
                if len(sent.strip().split()) <= length:
                    idx[num] = 1
            return idx

        dx = get_short_indices(max_len, sent1)
        guid = guid[dx]
        sent1 = sent1[dx]
        sent2 = sent2[dx]
        labels = labels[dx]

        dx = get_short_indices(max_len, sent2)
        guid = guid[dx]
        sent1 = sent1[dx]
        sent2 = sent2[dx]
        labels = labels[dx]

    else:
        # just trim first, so leaks do not contain the pad character
        sent1 = _trim_too_long(max_len, sent1)
        sent2 = _trim_too_long(max_len, sent2)

    # --------------------------------

    # encode labels
    if mode != Split.pred:
        label_list = processor.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}
        labels = np.array([label_map[ln] for ln in labels])

    # --------------------------------

    # tokenize
    sent1_tokenized = []
    for sent in sent1:
        li = []
        for word in sent.strip().split():
            if word in vocab_to_int.keys():
                li.append(vocab_to_int[word])
            else:
                li.append(vocab_to_int["<UNK>"])
        sent1_tokenized.append(li)

    sent2_tokenized = []
    for sent in sent2:
        li = []
        for word in sent.strip().split():
            if word in vocab_to_int.keys():
                li.append(vocab_to_int[word])
            else:
                li.append(vocab_to_int["<UNK>"])
        sent2_tokenized.append(li)

    # --------------------------------

    del sent1, sent2  # freeing up the memory
    gc.collect()

    # --------------------------------

    # Keeping track of common words
    leaks = [
        [len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
        for x1, x2 in zip(sent1_tokenized, sent2_tokenized)
    ]
    leaks = np.array(leaks)

    # --------------------------------

    # padding the sequences
    sent1_padded = pad_sequences(
        sent1_tokenized,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=vocab_to_int["<PAD>"],
    )
    sent2_padded = pad_sequences(
        sent2_tokenized,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=vocab_to_int["<PAD>"],
    )

    # --------------------------------

    del sent1_tokenized, sent2_tokenized  # freeing up the memory
    gc.collect()

    # --------------------------------

    return (sent1_padded, sent2_padded, leaks, labels)


def create_test_data(sent1, sent2, vocab_to_int, max_len):
    sent1 = clean_text(sent1)
    sent2 = clean_text(sent2)

    # --------------------------------

    sent1 = _trim_too_long(max_len, sent1)
    sent2 = _trim_too_long(max_len, sent2)

    # --------------------------------

    sent1_tokenized = []
    for sent in sent1:
        li = []
        for word in sent.strip().split():
            if word in vocab_to_int.keys():
                li.append(vocab_to_int[word])
            else:
                li.append(vocab_to_int["<UNK>"])
        sent1_tokenized.append(li)

    sent2_tokenized = []
    for sent in sent2:
        li = []
        for word in sent.strip().split():
            if word in vocab_to_int.keys():
                li.append(vocab_to_int[word])
            else:
                li.append(vocab_to_int["<UNK>"])
        sent2_tokenized.append(li)

    # --------------------------------

    del sent1, sent2  # freeing up the memory
    gc.collect()

    # --------------------------------

    # Keeping track of common words
    leaks = [
        [len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
        for x1, x2 in zip(sent1_tokenized, sent2_tokenized)
    ]
    leaks = np.array(leaks)

    # --------------------------------

    # padding the sequences
    sent1_padded = pad_sequences(
        sent1_tokenized,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=vocab_to_int["<PAD>"],
    )
    sent2_padded = pad_sequences(
        sent2_tokenized,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=vocab_to_int["<PAD>"],
    )

    # --------------------------------

    del sent1_tokenized, sent2_tokenized  # freeing up the memory
    gc.collect()

    # --------------------------------

    return (sent1_padded, sent2_padded, leaks)


# ---------------------------------------------------------------------------


def get_vocab(args: MyTrainingArguments):
    fn_cached_vocab = os.path.join(
        args.cache_dir, f"cached_vocab_{args.embedding_dim}.p"
    )
    if os.path.exists(fn_cached_vocab):
        if args.overwrite_cache:
            logger.info("Overwrite existing cache file.")
        else:
            with open(fn_cached_vocab, "rb") as fp:
                return pickle.load(fp)
    vocab_to_int, int_to_vocab, word_to_vec_map = read_glove_vectors(args.fn_vectors)
    vocab_to_int, int_to_vocab, word_to_vec_map = update_special_tokens(
        vocab_to_int, int_to_vocab, word_to_vec_map
    )

    with open(fn_cached_vocab, "wb") as fp:
        pickle.dump(
            (vocab_to_int, int_to_vocab, word_to_vec_map),
            fp,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return vocab_to_int, int_to_vocab, word_to_vec_map


def get_samples(args: MyTrainingArguments, mode: Split, vocab_to_int):
    fn_cached = os.path.join(
        args.cache_dir, f"cached_{mode.value}_{args.max_seq_length}.p"
    )

    if os.path.exists(fn_cached):
        logger.info(f"Cached samples exist: {fn_cached}")
        if args.overwrite_cache:
            logger.info("Overwrite existing cache file.")
        else:
            with open(fn_cached, "rb") as fp:
                return pickle.load(fp)

    sent1_padded, sent2_padded, leaks, labels = preprocess(
        fn_data=args.data_dir,
        vocab_to_int=vocab_to_int,
        max_len=args.max_seq_length,
        mode=mode,
        filter_too_long=args.discard_too_long_samples,
    )

    logger.info(f"Write cached samples to: {fn_cached}")
    with open(fn_cached, "wb") as fp:
        pickle.dump(
            (sent1_padded, sent2_padded, leaks, labels),
            fp,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return sent1_padded, sent2_padded, leaks, labels


# ---------------------------------------------------------------------------


def pretrained_embedding_layer(word_to_vec_map, words_to_index):
    emb_dim = word_to_vec_map["pen"].shape[0]
    vocab_size = len(words_to_index) + 1
    emb_matrix = np.zeros((vocab_size, emb_dim))

    for word, index in words_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # emb_matrix = tf.convert_to_tensor(emb_matrix, dtype=tf.float32)
    # emb_layer = Embedding(vocab_size, emb_dim, embeddings_initializer=Constant(emb_matrix), trainable=False)

    emb_layer = Embedding(vocab_size, emb_dim, trainable=True)
    emb_layer.build((None,))
    emb_layer.set_weights([emb_matrix])

    return emb_layer


# ---------------------------------------------------------------------------


def get_uncompiled_model(
    args: MyTrainingArguments, word_to_vec_map, vocab_to_int, leaks_dim: int = 3
):
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, vocab_to_int)
    lstm_layer1 = Bidirectional(
        LSTM(
            args.number_lstm,
            dropout=args.rate_drop_lstm,
            recurrent_dropout=args.rate_drop_lstm,
            return_sequences=True,
        )
    )
    lstm_layer2 = Bidirectional(
        LSTM(
            args.number_lstm,
            dropout=args.rate_drop_lstm,
            recurrent_dropout=args.rate_drop_lstm,
        )
    )
    dropout_layer = Dropout(0.5)

    seq1_inp = Input(shape=(args.max_seq_length,), dtype="int32", name="seq1_inp")
    net1 = embedding_layer(seq1_inp)
    net1 = lstm_layer1(net1)
    net1 = dropout_layer(net1)
    out1 = lstm_layer2(net1)

    seq2_inp = Input(shape=(args.max_seq_length,), dtype="int32", name="seq2_inp")
    net2 = embedding_layer(seq2_inp)
    net2 = lstm_layer1(net2)
    net2 = dropout_layer(net2)
    out2 = lstm_layer2(net2)

    # leaks.shape[1]
    leaks_inp = Input(shape=(leaks_dim,), name="leaks_inp")
    leaks_out = Dense(
        units=int(args.number_dense_units / 2), activation=args.activation_function
    )(leaks_inp)

    merged = concatenate([out1, out2, leaks_out])
    merged = BatchNormalization()(merged)
    merged = Dropout(args.rate_drop_dense)(merged)
    merged = Dense(args.number_dense_units, activation=args.activation_function)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(args.rate_drop_dense)(merged)

    preds = Dense(units=1, activation="sigmoid", name="pred")(merged)

    model = keras.models.Model(inputs=[seq1_inp, seq2_inp, leaks_inp], outputs=preds)

    return model


def get_compiled_model(
    args: MyTrainingArguments, word_to_vec_map, vocab_to_int, leaks_dim: int = 3
):
    model = get_uncompiled_model(
        args, word_to_vec_map, vocab_to_int, leaks_dim=leaks_dim
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


# ---------------------------------------------------------------------------


def data_generator(sent1, sent2, leaks, labels, batch_size):
    while True:
        idx = np.random.randint(len(sent1), size=batch_size)
        x1_batch = sent1[idx]
        x2_batch = sent2[idx]
        labels_batch = labels[idx]
        leaks_batch = leaks[idx]

        x_data = {"seq1_inp": x1_batch, "seq2_inp": x2_batch, "leaks_inp": leaks_batch}
        y_data = {"pred": labels_batch}

        yield (x_data, y_data)


# ---------------------------------------------------------------------------


def run(json_config_file=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(MyTrainingArguments)

    if json_config_file is not None:
        (args,) = parser.parse_json_file(json_file=json_config_file)
    else:
        (args,) = parser.parse_args_into_dataclasses()

    # args: MyTrainingArguments = args

    # -----------------------------------------------------------------------

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # -----------------------------------------------------------------------
    # Setup logging

    logging.basicConfig(
        format="[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(f"Parameters {args}")

    # -----------------------------------------------------------------------

    # Set seed
    set_seed(args.seed)

    # -----------------------------------------------------------------------

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir, exist_ok=True)

    # fn_best_model = os.path.join(args.output_dir, f"bestmodel_{args.max_seq_length}_{args.num_epochs}_{args.per_device_train_batch_size}.txt")
    fn_best_model = os.path.join(args.output_dir, "bestmodel.txt")
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints", str(int(time.time())))

    STAMP = (
        f"lstm_{args.number_lstm:d}_{args.number_dense_units:d}"
        "_{args.rate_drop_dense:.2f}_{args.rate_drop_lstm:.2f}"
    )
    bst_model_path = os.path.join(checkpoint_dir, f"{STAMP}.h5")

    # -----------------------------------------------------------------------
    # cache vocab and data

    vocab_to_int, _, word_to_vec_map = get_vocab(args)

    if args.do_train:
        _ = get_samples(args, Split.train, vocab_to_int)
        _ = get_samples(args, Split.dev, vocab_to_int)
    if args.do_eval:
        _ = get_samples(args, Split.dev, vocab_to_int)
    if args.do_test:
        _ = get_samples(args, Split.test, vocab_to_int)
    if args.do_predict:
        _ = get_samples(args, Split.pred, vocab_to_int)

    # cache is not fresh, do not overwrite again below ...
    args.overwrite_cache = False

    # -----------------------------------------------------------------------
    # model

    model = get_compiled_model(args, word_to_vec_map, vocab_to_int)

    if args.print_model:
        model.summary()

    # -----------------------------------------------------------------------
    # train

    results = dict()

    if args.do_train:
        vocab_to_int, _, _ = get_vocab(args)

        sent1, sent2, leaks, labels = get_samples(args, Split.train, vocab_to_int)
        sent1_valid, sent2_valid, leaks_valid, labels_valid = get_samples(
            args, Split.dev, vocab_to_int
        )

        # ----------------------
        # truncate

        logger.info(f"num train: {len(labels)}")
        logger.info(f"num dev:   {len(labels_valid)}")

        if False:
            idx_map = np.arange(len(labels))
            idx_map = np.random.choice(idx_map, 30000, replace=False)
            sent1, sent2, leaks, labels = (
                sent1[idx_map],
                sent2[idx_map],
                leaks[idx_map],
                labels[idx_map],
            )
            logger.info(f"num train: {len(labels)}")

        if True:
            logger.warn("Truncating dev dataset!")
            idx_map = np.arange(len(labels_valid))
            idx_map = np.random.choice(idx_map, 10000, replace=False)
            sent1_valid, sent2_valid, leaks_valid, labels_valid = (
                sent1_valid[idx_map],
                sent2_valid[idx_map],
                leaks_valid[idx_map],
                labels_valid[idx_map],
            )
            logger.info(f"num dev:   {len(labels_valid)}")

        # ----------------------

        labels = labels.astype(float)
        labels_valid = labels_valid.astype(float)

        # ----------------------

        early_stopping = EarlyStopping(monitor="val_loss", patience=7)

        os.makedirs(checkpoint_dir, exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            bst_model_path, save_best_only=True, save_weights_only=False
        )
        tensorboard = TensorBoard(
            log_dir=os.path.join(checkpoint_dir, "logs", f"{time.time()}")
        )

        # ----------------------

        generator = data_generator(
            sent1, sent2, leaks, labels, args.per_device_train_batch_size
        )

        # ----------------------

        NUM_TRAIN_SAMPLES = len(labels)
        NUM_EVAL_SAMPLES = len(labels_valid)

        logger.info(f"number of train samples: {NUM_TRAIN_SAMPLES}")
        logger.info(f"number of eval samples:  {NUM_EVAL_SAMPLES}")
        logger.info(f"batch size:              {args.per_device_eval_batch_size}")
        logger.info(f"number of epochs:        {args.num_epochs}")
        logger.info(
            f"steps per train epoch:   {(NUM_TRAIN_SAMPLES // args.per_device_eval_batch_size)}"
        )

        # NOTE: continue training too complicated ...

        H = model.fit(
            x=generator,
            steps_per_epoch=NUM_TRAIN_SAMPLES // args.per_device_eval_batch_size,
            epochs=args.num_epochs,
            callbacks=[
                early_stopping,
                model_checkpoint,
                tensorboard,
                # TqdmCallback(verbose=0),
            ],
            # verbose=0,
            validation_data=([sent1_valid, sent2_valid, leaks_valid], labels_valid),
            validation_batch_size=args.per_device_eval_batch_size,
            # validation_steps=NUM_EVAL_SAMPLES // args.per_device_eval_batch_size,
        )

        # H = model.fit_generator(
        #     generator,
        #     steps_per_epoch=80,
        #     epochs=NUM_EPOCHS,
        #     callbacks=[
        #         early_stopping,
        #         model_checkpoint,
        #         tensorboard
        #     ],
        #     validation_data=(
        #         [sent1_valid, sent2_valid, leaks_valid],
        #         labels_valid
        #     )
        # )

        with open(fn_best_model, "w") as file:
            file.write(bst_model_path)

        # ----------------------

        N = min(args.per_device_train_batch_size, len(H.history["loss"]))

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(args.output_dir, "plot.png"))

        # ----------------------

        results["train"] = {
            "loss": H.history["loss"],
            "accuracy": H.history["accuracy"],
            **(
                {}
                if "val_loss" not in H.history
                else {
                    "val_loss": H.history["val_loss"],
                    "val_accuracy": H.history["val_accuracy"],
                }
            ),
        }

        with open(os.path.join(args.output_dir, "train_stats.json"), "w") as fp:
            fp.write(json.dumps(results))

    # -----------------------------------------------------------------------
    # dev/test/pred

    def _find_model(check_path):
        if os.path.isfile(check_path) and str(check_path).endswith(".h5"):
            return check_path

        if os.path.basename(check_path) == "checkpoints":
            check_path = os.path.dirname(check_path)

        fn_best_model = os.path.join(check_path, "bestmodel.txt")
        if os.path.isfile(fn_best_model):
            with open(fn_best_model, "r") as file:
                return file.read()

        # try to find last checkpoint

        check_path = os.path.join(check_path, "checkpoints")
        if not os.path.isdir(check_path):
            return None

        content = os.listdir(check_path)
        checkpoints = [
            path
            for path in content
            if path.isdigit() and os.path.isdir(os.path.join(check_path, path))
        ]
        if not checkpoints:
            return None

        checkpoint_dir = os.path.join(
            check_path, max(checkpoints, key=lambda x: int(x))
        )
        if not os.path.isdir(checkpoint_dir):
            return None

        bst_model_path = os.path.join(checkpoint_dir, f"{STAMP}.h5")
        if not os.path.isfile(bst_model_path):
            return None

        return bst_model_path

    def _load_model(args: MyTrainingArguments):
        best_model_path = None

        if args.model_name_or_path is not None:
            logger.info(f"Try loading model from path: {args.model_name_or_path}")
            best_model_path = _find_model(args.model_name_or_path)
            if best_model_path:
                logger.info(f"Found model in path: {best_model_path}")

        if not best_model_path:
            fn_best_model = os.path.join(args.output_dir, "bestmodel.txt")
            with open(fn_best_model, "r") as file:
                best_model_path = file.read()
            logger.info(f"Load model path from best model: {best_model_path}")

        model = keras.models.load_model(best_model_path)

        return model

    def _run_preds(mode: Split):
        model = _load_model(args)

        # ----------------------

        vocab_to_int, _, _ = get_vocab(args)

        sent1_test, sent2_test, leaks_test, labels_test = get_samples(
            args, mode, vocab_to_int
        )

        # ----------------------

        preds_test = model.predict(
            x=[sent1_test, sent2_test, leaks_test],
            verbose=1,
            batch_size=args.per_device_eval_batch_size,
        )

        preds_test = np.squeeze(preds_test)

        return preds_test, labels_test

    def _binarize_preds(preds):
        # preds[preds <= 0.5] = 0.
        # preds[preds > 0.5] = 1.
        # preds = preds.astype("int32")
        preds = (preds > 0.5).astype("int32")
        return preds

    def _store_preds(mode: Split, preds):
        output_preds_file = os.path.join(args.output_dir, f"{mode.value!s}_results.txt")
        with open(output_preds_file, "w") as writer:
            writer.write("index\tprediction\n")
            for index, item in enumerate(preds):
                writer.write(f"{index}\t{item}\n")

    def _store_metrics(mode: Split, preds, y_true, precision=8):
        output_metrics_file = os.path.join(
            args.output_dir, f"{mode.value!s}_metrics.json"
        )

        metrics = compute_metrics(y_true, preds, precision=precision)

        with open(output_metrics_file, "w") as fp:
            json.dump(metrics, fp, indent=4, cls=NumpyEncoder)

    # -----------------------------------------------------------------------
    # dev/test/pred

    if args.do_eval:
        preds_dev, labels_dev = _run_preds(Split.dev)

        preds_dev = _binarize_preds(preds_dev)

        _store_preds(Split.dev, preds_dev)
        _store_metrics(Split.dev, preds_dev, labels_dev)

        print(f"***** {str(Split.dev.value).title()} results *****")
        print(
            classification_report(
                labels_dev, preds_dev, digits=4, target_names=["not same", "same"]
            )
        )

    if args.do_test:
        preds_test, labels_test = _run_preds(Split.test)

        preds_test = _binarize_preds(preds_test)

        _store_preds(Split.test, preds_test)
        _store_metrics(Split.test, preds_test, labels_test)

        # ----------------------

        print(f"***** {str(Split.test.value).title()} results *****")
        print(
            classification_report(
                labels_test, preds_test, digits=4, target_names=["not same", "same"]
            )
        )

    if args.do_predict:
        predictions, _ = _run_preds(Split.pred)
        _store_preds(Split.pred, predictions)

    # -----------------------------------------------------------------------

    return results


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    json_config_file = None
    run(json_config_file=json_config_file)
