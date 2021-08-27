import gzip
import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# data loading


def load_ds(fn):
    def _loader(fn):
        with gzip.open(fn, "rb") as fp:
            for line in fp:
                yield json.loads(line)

    def _cleaner(it):
        for entry in it:
            entry = {k: v for k, v in entry.items() if k in ("reviewerID", "asin", "reviewText", "overall")}
            yield entry

    it = _loader(fn)
    it = _cleaner(it)

    df = pd.DataFrame.from_dict(it)
    df.rename(columns={"reviewerID": "rid", "asin": "id", "reviewText": "text", "overall": "rating"}, inplace=True)

    df["goodness"] = df["rating"] >= 4.0
    # df["goodness"].value_counts()

    return df


# ---------------------------------------------------------------------------


def load_all_ds(fn_root):
    fn_root = Path(fn_root)

    all_ds = list()
    for fn_ds in sorted(fn_root.iterdir()):
        name = fn_ds.name
        if not name.startswith("reviews_") or not name.endswith("_5.json.gz"):
            continue

        topic = name.split("_", 1)[-1].rsplit("_", 1)[0]

        print(f"Load reviews for {topic} ...")
        df_ds = load_ds(fn_ds)
        df_ds["topic"] = topic
        all_ds.append(df_ds)

    df = pd.concat(all_ds)
    df = df.reset_index()  # !important to have unique .index values
    return df


# ---------------------------------------------------------------------------


def make_inv_topic2id(df):
    df = df[["id", "topic"]].drop_duplicates()
    inv_topic2id = {topic: df_t.to_list() for topic, df_t in df.groupby("topic")["id"]}
    return inv_topic2id


def make_inv_id2topic(inv_topic2id):
    return {id_: topic for topic, ids in inv_topic2id.items() for id_ in ids}


# ---------------------------------------------------------------------------
