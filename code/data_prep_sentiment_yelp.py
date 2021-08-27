import json
import os
import pickle
from collections import Counter
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm

try:
    from transformers.trainer_utils import set_seed
except ImportError:
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass

from utils import Timer

tqdm.pandas()

# ---------------------------------------------------------------------------

# download + scp to server + extract
data_yelp_path = Path("data/sentiment/yelp/")

# ------------------------------------

# local?
data_yelp_path = Path("data_raw/sentiment/yelp/")

# local? - output path (base) for sentiment review yelp pairs
data_yelp_b_tdt_path = Path("data/sentiment/yelp-pair-b/")
# local - output path for simple sentiment reviews yelp
data_yelp_tdt_sentiment_5_path = Path("data/sentiment/yelp-sentiment-5/")
data_yelp_tdt_sentiment_b_path = Path("data/sentiment/yelp-sentiment-b/")

# ---------------------------------------------------------------------------
# load categories and topics


def load_reviews(fn_yelp_reviews):
    """Load Yelp reviews. Return a Pandas dataframe.
    Format: {"id": business_id, "rid": review_id, "text": text, "rating": rating}
    And add goodness bool (rating > 3)."""
    data = list()

    with jsonlines.open(fn_yelp_reviews, "r") as fp:
        for n, entry in enumerate(tqdm(fp)):
            # print(entry)
            # break
            business_id = entry.pop("business_id")
            review_id = entry.pop("review_id")
            text = entry.pop("text", "")
            rating = entry.pop("stars", 0.0)
            data.append({"id": business_id, "rid": review_id, "text": text, "rating": rating})
            # TESTING
            # if n > 10000:
            #     break

    df = pd.DataFrame.from_dict(data)
    
    # Add goodness value
    # TODO: maybe ignore with 3
    df["goodness"] = df["rating"] > 3
    
    return df


def load_topics(fn_yelp_topics, bids_not_cats=None, filter_cats=None, filter_cat_combis=None):
    """Load topics (categories).
    Optionally filter by giving it a whitelist of allowed categories.
    Optionally filter by giving a list of allowed category combinations.
    Optionally provide an IN/OUT param ``bids_not_cats`` that accumulates business ids without categories."""
    inv_bid_cats = dict()
    if bids_not_cats is None or not isinstance(bids_not_cats, set):
        bids_not_cats = set()
    
    # load lookup business -> categories
    with jsonlines.open(fn_yelp_topics, "r") as fp:
        for n, entry in enumerate(tqdm(fp)):
            business_id = entry.pop("business_id")
            categories = entry.pop("categories")

            if not categories:
                bids_not_cats.add(business_id)
                continue

            categories = categories.split(", ")
            
            if filter_cats:
                categories = [c for c in categories if c in filter_cats]
                if not categories:
                    # bids_not_cats.add(business_id)  # ??
                    continue
                    
            if filter_cat_combis:
                # skip if combination is not above threshold (of filter list)
                if tuple(sorted(set(categories))) not in filter_cat_combis:
                    continue

            inv_bid_cats[business_id] = categories
    
    return inv_bid_cats


# ---------------------------------------------------------------------------
# Filter categories

def filter_min_cat_combis(inv_cat_combis, min_num=30):
    """Filter category combinations by minimum amount of occurrences in businesses"""
    f_inv_cat_combis = dict()
    
    for cats, num in inv_cat_combis.items():
        if num >= min_num:
            f_inv_cat_combis[cats] = num
    
    return f_inv_cat_combis


def make_map_cats(inv_bid_cats):
    """Make a map from category to business id"""
    inv_cat_bids = dict()

    # reverse lookup: category -> businesses
    for bid, cats in tqdm(inv_bid_cats.items()):
        for cat in cats:
            try:
                inv_cat_bids[cat].append(bid)
            except KeyError:
                inv_cat_bids[cat] = [bid]
                
    ## TODO: make distinct?
    for cat in inv_cat_bids.keys():
        inv_cat_bids[cat] = list(set(inv_cat_bids[cat]))
                
    return inv_cat_bids


def make_cat_combis(inv_bid_cats):
    """Count amount of each category combination occurring in businesses"""
    inv_cat_combis = Counter()

    inv_cat_combis.update(
        (tuple(sorted(set(cats))) for cats in tqdm(inv_bid_cats.values())))
    
    return inv_cat_combis


# ---------------------------------------------------------------------------
# Filter reviews


def filter_min_review_freq(df, min_ratings=5):
    """Filter review dataframe for a minimum of N of each good and bad ratings."""
    # filter with at least N ratings per goodness
    df_filter = df.groupby(["id", "goodness"])[["id"]].count() < min_ratings
    df_filter = df_filter.rename(columns={"id": "filter"})
    df_filter = df_filter[df_filter["filter"] == True]

    # build a filter id list
    df_filter_list = df_filter.reset_index()["id"].to_list()

    # filter with list
    df_filtered = df[~df.id.isin(df_filter_list)]
    
    return df_filtered


def filter_both_good_bad(df):
    """Filter the dataframe to contain only both good and bad reviews for each business.
    Dataframe should be the same if minimum filtering above is done."""
    # build filter for ids that contain both positive and negative samples
    df_filter = df.groupby(["id", "goodness"], as_index=False).count().groupby("id")[["id"]].count() == 2
    df_filter = df_filter.rename(columns={"id": "filter"})
    df_filter = df_filter[df_filter["filter"] == True]

    # create list of IDs for which this is true
    df_filter_list = df_filter.reset_index()["id"].to_list()

    # filter with list
    df_filtered = df[df.id.isin(df_filter_list)]

    # df_filtered.groupby(["id", "goodness"]).count()
    return df_filtered


# ---------------------------------------------------------------------------
# Filter businesses


def filter_by_businesses(df, lst_business_ids):
    # filter with list, keep businesses in list
    df_filtered = df[df.id.isin(set(lst_business_ids))]

    return df_filtered


def filter_by_businesses_not_same(df, lst_business_ids):
    # filter with list, keep businesses that are not in list
    df_filtered = df[~df.id.isin(set(lst_business_ids))]

    return df_filtered


# ---------------------------------------------------------------------------
# Filter other category businesses


def filter_root_category_businesses_uniq(dn_yelp_cached, category_label, inv_cat_bids, map_categories, map_cat_name2id):
    df_root_cat = load_cached_root_category_businesses_df(dn_yelp_cached, category_label, map_categories)
    
    root_categories = get_root_category_items(map_categories)
    # root_categories = sorted(root_categories, key=lambda x: len(x["businesses"]), reverse=False)
    
    for root_category in root_categories:
        if root_category["title"] == category_label:
            # skip, do not trim self
            continue
            
        business_ids = set(get_businesses_in_category_branch(inv_cat_bids, root_category["title"], map_categories, map_cat_name2id))
        # business_ids = set(root_category["businesses"])
        print(f"""Filter businesses from category {root_category["title"]} [{root_category["alias"]}] ({len(business_ids)} businesses) ...""")
        n_before = len(df_root_cat)
        df_root_cat = filter_by_businesses_not_same(df_root_cat, business_ids)
        n_after = len(df_root_cat)
        print(f"""Filtered {n_before - n_after} businesses (overlap with {root_category["title"]})""")
        
    return df_root_cat


def filter_root_category_businesses_not_other(dn_yelp_cached, category_label, category_label_filter, inv_cat_bids, map_categories, map_cat_name2id):
    assert category_label != category_label_filter, "do not filter on self"

    df_root_cat = load_cached_root_category_businesses_df(dn_yelp_cached, category_label, map_categories)
    
    root_categories = get_root_category_items(map_categories)
    # root_categories = sorted(root_categories, key=lambda x: len(x["businesses"]), reverse=True)
    
    for root_category in root_categories:
        if root_category["title"] == category_label_filter:
            break
    else:
        print(f"No businesses found for {category_label_filter} -> return unchanged")
        return df_root_cat
    
    business_ids = set(get_businesses_in_category_branch(inv_cat_bids, root_category["title"], map_categories, map_cat_name2id))
    # businesses_ids = root_category["businesses"]
    print(f"""Filter businesses from category {root_category["title"]} [{root_category["alias"]}] ({len(set(business_ids))} businesses) ...""")
    n_before = len(df_root_cat)
    df_root_cat = filter_by_businesses_not_same(df_root_cat, business_ids)
    n_after = len(df_root_cat)
    print(f"""Filtered {n_before - n_after} businesses (overlap with {root_category["title"]})""")
    
    return df_root_cat


def filter_root_category_businesses_same_other(dn_yelp_cached, category_label, category_label_filter, inv_cat_bids, map_categories, map_cat_name2id):
    assert category_label != category_label_filter, "do not filter on self"

    df_root_cat = load_cached_root_category_businesses_df(dn_yelp_cached, category_label, map_categories)
    
    root_categories = get_root_category_items(map_categories)
    # root_categories = sorted(root_categories, key=lambda x: len(x["businesses"]), reverse=True)
    
    for root_category in root_categories:
        if root_category["title"] == category_label_filter:
            break
    else:
        print(f"No businesses found for {category_label_filter} -> return unchanged")
        return df_root_cat
    
    business_ids = set(get_businesses_in_category_branch(inv_cat_bids, root_category["title"], map_categories, map_cat_name2id))
    # businesses_ids = root_category["businesses"]
    print(f"""Filter businesses from category {root_category["title"]} [{root_category["alias"]}] ({len(set(business_ids))} businesses) ...""")
    n_before = len(df_root_cat)
    df_same = filter_by_businesses(df_root_cat, business_ids)
    n_after = len(df_same)
    print(f"""Filtered {n_before - n_after} businesses ({n_after} same with {root_category["title"]})""")
    
    return df_same


# ---------------------------------------------------------------------------
# Load category tree


def load_category_tree(fn_all_category_list):
    with open(fn_all_category_list, "r") as fp:
        content = fp.read()
        data = json.loads(content)

    map_categories = dict()
    map_cat_name2id = dict()
    lst_root_categories = list()

    # load basic lookups
    for item in data:
        # .alias (id)
        map_categories[item["alias"]] = item
        # .title
        map_cat_name2id[item["title"]] = item["alias"]
        # .parents
        # some have multiple parents ...
        if not item["parents"]:
            lst_root_categories.append(item["alias"])
        # add list of children
        item["children"] = list()

    # add children
    for cid, item in map_categories.items():
        for parent_cid in item["parents"]:
            map_categories[parent_cid]["children"].append(item["alias"])

    return map_categories, map_cat_name2id, lst_root_categories


def get_root_category_items(map_categories):
    lst_root_categories = list()

    for cid, item in map_categories.items():
        if not item["parents"]:
            lst_root_categories.append(item)
            
    return lst_root_categories


def get_children_category_item_list(map_categories, parent_cid):
    return [
        map_categories[child_cid]
        for child_cid in map_categories[parent_cid]["children"]
    ]


def get_businesses_in_category(inv_cat_bids, category):
    try:
        return list(set(inv_cat_bids[category]))
    except KeyError:
        return []
    
    
def get_businesses_in_category_branch(inv_cat_bids, category, map_categories, map_cat_name2id):
    map_cat_id2name = {cid: name for name, cid in map_cat_name2id.items()}

    def _get_recursive_businesses(cat_name):
        businesses = get_businesses_in_category(inv_cat_bids, cat_name)

        cid = map_cat_name2id[cat_name]
        for child_cid in map_categories[cid]["children"]:
            child_name = map_cat_id2name[child_cid]
            businesses.extend(_get_recursive_businesses(child_name))
        
        return businesses
    
    return _get_recursive_businesses(category)


def get_reviews_for_category(df, cat_name, inv_cat_bids, map_categories, map_cat_name2id):
    businesses = get_businesses_in_category_branch(inv_cat_bids, cat_name, map_categories, map_cat_name2id)
    print(f"""{cat_name}: {len(businesses)}, uniq: {len(set(businesses))}""")
    businesses = set(businesses)

    df_businesses = filter_by_businesses(df, businesses)

    return df_businesses


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def cache_root_category_businesses_df(df, inv_cat_bids, map_categories, map_cat_name2id):
    root_categories = get_root_category_items(map_categories)
    root_categories = sorted(root_categories, key=lambda x: x["title"])
    
    dn_yelp_cached = data_yelp_path / "cached"
    if not dn_yelp_cached.exists():
        print(f"Create cache dir: {dn_yelp_cached}")
        dn_yelp_cached.mkdir()
        
    for root_category in root_categories:
        fn_yelp_cached_root_cat_df = dn_yelp_cached / f"""{root_category["alias"]}_businesses.df.p"""
        if fn_yelp_cached_root_cat_df.exists():
            continue

        business_ids = set(get_businesses_in_category_branch(inv_cat_bids, root_category["title"], map_categories, map_cat_name2id))
        # business_ids = set(root_category["businesses"])
        print(f"""Filter category {root_category["title"]} [{root_category["alias"]}] with {len(set(business_ids))} businesses ...""")
        df_root_cat = filter_by_businesses(df, business_ids)

        # df_root_cat = get_reviews_for_category(df, cat_name, inv_cat_bids, map_categories, map_cat_name2id)

        df_root_cat.to_pickle(str(fn_yelp_cached_root_cat_df))
        
        
def load_cached_root_category_businesses_df(dn_yelp_cached, category_label, map_categories):
    root_categories = get_root_category_items(map_categories)
    root_categories = sorted(root_categories, key=lambda x: x["title"])

    for root_category in root_categories:
        if root_category["title"] == category_label:
            category_id = root_category["alias"]
            break
    else:
        print(f"No cached root category businesses found for: {category_label}")
        return None

    fn_yelp_cached_root_cat_df = dn_yelp_cached / f"{category_id}_businesses.df.p"
    if not fn_yelp_cached_root_cat_df.exists():
        return None
    
    df_root_cat = pd.read_pickle(str(fn_yelp_cached_root_cat_df))
    return df_root_cat


# ---------------------------------------------------------------------------
# Make Pairs
# ---------------------------------------------------------------------------
# Positive + negative same-sentiment pairs


def make_pairs_good_bad(df, inv_bid_cats, num_pairs_per_class=2):
    pairs_good = list()
    pairs_bad = list()

    for id_, group in tqdm(df.groupby("id")):
        grouper = group.groupby("goodness")
        reviews_good = grouper.get_group(True)
        reviews_bad = grouper.get_group(False)

        # TESTING
        # print("id:", id_)
        # print("#good:", len(reviews_good))
        # print("#bad:", len(reviews_bad))
        # print(group)
        # break

        # make pairings -- good ss
        rg_idx = reviews_good.index.values
        # print("pos_idx:", rg_idx)
        rg_idx_sel = np.random.choice(rg_idx, 2 * num_pairs_per_class, replace=False)
        for id1, id2 in zip(rg_idx_sel[::2], rg_idx_sel[1::2]):
            # print("pair:", id1, id2)
            r1, r2 = df.loc[id1], df.loc[id2]
            pair = {
                "argument1": r1["text"], "argument2": r2["text"],
                "argument1_id": f"""{r1["id"]}|{r1["rid"]}""", "argument2_id": f"""{r2["id"]}|{r2["rid"]}""",
                "is_same_side": True, "is_good_side": True,
                "type": "good-good",
                "topic": inv_bid_cats.get(r1["id"], None)
            }
            # print(pair)
            pairs_good.append(pair)

        # make pairings -- bad ss
        rb_idx = reviews_bad.index.values
        # print("neg_idx:", rb_idx)
        rb_idx_sel = np.random.choice(rb_idx, 2 * num_pairs_per_class, replace=False)
        for id1, id2 in zip(rb_idx_sel[::2], rb_idx_sel[1::2]):
            r1, r2 = df.loc[id1], df.loc[id2]
            pair = {
                "argument1": r1["text"], "argument2": r2["text"],
                "argument1_id": f"""{r1["id"]}|{r1["rid"]}""", "argument2_id": f"""{r2["id"]}|{r2["rid"]}""",
                "is_same_side": True, "is_good_side": False,
                "type": "bad-bad",
                "topic": inv_bid_cats.get(r1["id"], None)
            }
            pairs_bad.append(pair)

        # break
        
    return pairs_good, pairs_bad


def make_pairs_good_bad_over_business(df, inv_bid_cats, num_pairs_per_side):
    pairs_good = list()
    pairs_bad = list()

    grouper = df.groupby("goodness")
    reviews_good = grouper.get_group(True)
    reviews_bad = grouper.get_group(False)

    # make pairings -- good ss
    rg_idx = reviews_good.index.values
    # print("pos_idx:", rg_idx)
    rg_idx_sel = np.random.choice(rg_idx, 2 * num_pairs_per_side, replace=False)
    for id1, id2 in zip(rg_idx_sel[::2], rg_idx_sel[1::2]):
        # print("pair:", id1, id2)
        r1, r2 = df.loc[id1], df.loc[id2]
        pair = {
            "argument1": r1["text"], "argument2": r2["text"],
            "argument1_id": f"""{r1["id"]}|{r1["rid"]}""", "argument2_id": f"""{r2["id"]}|{r2["rid"]}""",
            "is_same_side": True, "is_good_side": True,
            "type": "good-good",
            "topic": ", ".join(str(x) for x in [inv_bid_cats.get(r1["id"], None), inv_bid_cats.get(r2["id"], None)])
        }
        # print(pair)
        pairs_good.append(pair)

    # make pairings -- bad ss
    rb_idx = reviews_bad.index.values
    # print("neg_idx:", rb_idx)
    rb_idx_sel = np.random.choice(rb_idx, 2 * num_pairs_per_side, replace=False)
    for id1, id2 in zip(rb_idx_sel[::2], rb_idx_sel[1::2]):
        r1, r2 = df.loc[id1], df.loc[id2]
        pair = {
            "argument1": r1["text"], "argument2": r2["text"],
            "argument1_id": f"""{r1["id"]}|{r1["rid"]}""", "argument2_id": f"""{r2["id"]}|{r2["rid"]}""",
            "is_same_side": True, "is_good_side": False,
            "type": "bad-bad",
            "topic": ", ".join(str(x) for x in [inv_bid_cats.get(r1["id"], None), inv_bid_cats.get(r2["id"], None)])
        }
        pairs_bad.append(pair)
        
    return pairs_good, pairs_bad


# ---------------------------------------------------------------------------
# Not same-sentiment pairs (combinations positive + negative)


def make_pairs_negative(df, inv_bid_cats, num_pairs_negative, repeatable_on_side=False):
    pairs_not_ss = list()

    for id_, group in tqdm(df.groupby("id")):
        grouper = group.groupby("goodness")
        reviews_good = grouper.get_group(True)
        reviews_bad = grouper.get_group(False)

        # find indices for reviews per business
        rg_idx = reviews_good.index.values
        rb_idx = reviews_bad.index.values

        # randomly select from each side
        rg_idx_sel = np.random.choice(rg_idx, num_pairs_negative, replace=repeatable_on_side)
        rb_idx_sel = np.random.choice(rb_idx, num_pairs_negative, replace=repeatable_on_side)

        # pair them together -- good-bad pairs
        for idg, idb in zip(rg_idx_sel[::2], rb_idx_sel[::2]):
            rg, rb = df.loc[idg], df.loc[idb]
            pair = {
                "argument1": rg["text"], "argument2": rb["text"],
                "argument1_id": f"""{rg["id"]}|{rg["rid"]}""", "argument2_id": f"""{rb["id"]}|{rb["rid"]}""",
                "is_same_side": False, "is_good_side": None,
                "type": "good-bad",
                "topic": inv_bid_cats.get(rg["id"], None)
            }
            # print(pair)
            pairs_not_ss.append(pair)

        # bad-good pairs
        for idb, idg in zip(rb_idx_sel[1::2], rg_idx_sel[1::2]):
            rb, rg = df.loc[idb], df.loc[idg]
            pair = {
                "argument1": rb["text"], "argument2": rg["text"],
                "argument1_id": f"""{rb["id"]}|{rb["rid"]}""", "argument2_id": f"""{rg["id"]}|{rg["rid"]}""",
                "is_same_side": False, "is_good_side": None,
                "type": "bad-good",
                "topic": inv_bid_cats.get(rb["id"], None)
            }
            # print(pair)
            pairs_not_ss.append(pair)
            
    return pairs_not_ss


def make_pairs_negative_over_business(df, inv_bid_cats, num_pairs_per_side, repeatable_on_side=False):
    pairs_not_ss = list()

    grouper = df.groupby("goodness")
    reviews_good = grouper.get_group(True)
    reviews_bad = grouper.get_group(False)

    # find indices for reviews per business
    rg_idx = reviews_good.index.values
    rb_idx = reviews_bad.index.values

    # randomly select from each side
    rg_idx_sel = np.random.choice(rg_idx, 2 * num_pairs_per_side, replace=repeatable_on_side)
    rb_idx_sel = np.random.choice(rb_idx, 2 * num_pairs_per_side, replace=repeatable_on_side)

    # pair them together -- good-bad pairs
    for idg, idb in zip(rg_idx_sel[::2], rb_idx_sel[::2]):
        rg, rb = df.loc[idg], df.loc[idb]
        pair = {
            "argument1": rg["text"], "argument2": rb["text"],
            "argument1_id": f"""{rg["id"]}|{rg["rid"]}""", "argument2_id": f"""{rb["id"]}|{rb["rid"]}""",
            "is_same_side": False, "is_good_side": None,
            "type": "good-bad",
            "topic": ", ".join(str(x) for x in [inv_bid_cats.get(rg["id"], None), inv_bid_cats.get(rb["id"], None)])
        }
        # print(pair)
        pairs_not_ss.append(pair)

    # bad-good pairs
    for idb, idg in zip(rb_idx_sel[1::2], rg_idx_sel[1::2]):
        rb, rg = df.loc[idb], df.loc[idg]
        pair = {
            "argument1": rb["text"], "argument2": rg["text"],
            "argument1_id": f"""{rb["id"]}|{rb["rid"]}""", "argument2_id": f"""{rg["id"]}|{rg["rid"]}""",
            "is_same_side": False, "is_good_side": None,
            "type": "bad-good",
            "topic": ", ".join(str(x) for x in [inv_bid_cats.get(rg["id"], None), inv_bid_cats.get(rb["id"], None)])
        }
        # print(pair)
        pairs_not_ss.append(pair)
            
    return pairs_not_ss


# ---------------------------------------------------------------------------
# Dataframe for training etc.


def make_or_load_pairs(df, inv_bid_cats, fn_yelp_df, num_pairs_per_class=2, random_state=42):
    if fn_yelp_df is not None:
        if os.path.exists(fn_yelp_df):
            with open(fn_yelp_df, "rb") as fp:
                all_df = pickle.load(fp)
            return all_df
    
    pairs_good, pairs_bad = make_pairs_good_bad(df, inv_bid_cats, num_pairs_per_class=num_pairs_per_class)
    print("#ss (pos)", len(pairs_good))
    print("#ss (neg)", len(pairs_bad))
    
    num_pairs_negative = 2 * num_pairs_per_class
    pairs_not_ss = make_pairs_negative(df, inv_bid_cats, num_pairs_negative, repeatable_on_side=False)
    print("#nss", len(pairs_not_ss))
    
    pairs_all = pairs_good + pairs_bad + pairs_not_ss
    print("#~ss", len(pairs_all))
    
    pairs_all = shuffle(pairs_all, random_state=random_state)
    df_traindev = pd.DataFrame.from_dict(pairs_all)
    
    if fn_yelp_df is not None:
        with open(fn_yelp_df, "wb") as fp:
            pickle.dump(df_traindev, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    return df_traindev


def make_or_load_pairs_over_businesses(df, inv_bid_cats, fn_yelp_df, random_state=42):
    if fn_yelp_df is not None:
        if os.path.exists(fn_yelp_df):
            with open(fn_yelp_df, "rb") as fp:
                all_df = pickle.load(fp)
            return all_df

    num_businesses = sum(1 for _ in df.groupby("id"))
    num_pairs = num_businesses
    
    pairs_good, pairs_bad = make_pairs_good_bad_over_business(df, inv_bid_cats, num_pairs)
    print("#ss (pos)", len(pairs_good))
    print("#ss (neg)", len(pairs_bad))

    pairs_not_ss = make_pairs_negative_over_business(df, inv_bid_cats, num_pairs, repeatable_on_side=False)
    print("#nss", len(pairs_not_ss))
    
    pairs_all = pairs_good + pairs_bad + pairs_not_ss
    print("#~ss", len(pairs_all))
    
    pairs_all = shuffle(pairs_all, random_state=random_state)
    df_traindev = pd.DataFrame.from_dict(pairs_all)
    
    if fn_yelp_df is not None:
        with open(fn_yelp_df, "wb") as fp:
            pickle.dump(df_traindev, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
    return df_traindev


# ---------------------------------------------------------------------------
# train/dev/test
# ---------------------------------------------------------------------------
# train/dev/test split


def split_df(df, ratio=0.1, do_shuffle=True, random_state=42, name_train="train", name_dev="dev"):
    if do_shuffle:
        df = shuffle(df, random_state=random_state)

    # ------------------------------------

    assert ratio > 0.0 and ratio < 1.0

    num = len(df)
    idx_ratio = int(num * (1.0 - ratio))
    print(f"{name_train}: [0:{idx_ratio}], {name_dev}: [{idx_ratio}:{num}], ratio: {ratio}")

    train_df = df.iloc[:idx_ratio]
    dev_df = df.iloc[idx_ratio:]

    return train_df, dev_df


def write_pair_df_tsv(df, fn_tsv, desc="train", create=True):
    fn_tsv = Path(fn_tsv)
    if create:
        fn_tsv.parent.mkdir(parents=True, exist_ok=True)

    with fn_tsv.open("w", encoding="utf-8") as fp:
        for row in tqdm(df.itertuples(), desc=desc, total=len(df)):
            text1 = row.argument1.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
            text2 = row.argument2.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
            label = "1" if row.is_same_side else "0"
            fp.write(f"""{row.Index}\t{text1}\t{text2}\t{label}\n""")


def write_single_5_df_tsv(df, fn_tsv, desc="train"):
    fn_tsv = Path(fn_tsv)
    with fn_tsv.open("w", encoding="utf-8") as fp:
        for row in tqdm(df.itertuples(), desc=desc, total=len(df)):
            text = row.text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
            label = int(row.rating)
            fp.write(f"""{row.rid}\t{label}\t{text}\n""")


def write_single_2_df_tsv(df, fn_tsv, desc="train"):
    fn_tsv = Path(fn_tsv)
    with fn_tsv.open("w", encoding="utf-8") as fp:
        for row in tqdm(df.itertuples(), desc=desc, total=len(df)):
            text = row.text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
            label = "1" if row.goodness else "0"
            fp.write(f"""{row.rid}\t{label}\t{text}\n""")


def write_pair_tdt_tsv(root_path, traindev_df, split_test=0.1, split_dev=0.3, do_shuffle=True, random_state=42):
    root_path = Path(root_path)
    if not root_path.exists():
        print(f"Create dir: {root_path}")
        root_path.mkdir(parents=True)

    fn_train_tsv = root_path / "train.tsv"
    fn_dev_tsv = root_path / "dev.tsv"
    fn_test_tsv = root_path / "test.tsv"

    # ------------------------------------

    if do_shuffle:
        traindev_df = shuffle(traindev_df, random_state=random_state)

    # ------------------------------------

    if split_test is not None:
        traindev_df, test_df = split_df(traindev_df, ratio=split_test, do_shuffle=False, name_train="traindev", name_dev="test")

    train_df, dev_df = split_df(traindev_df, ratio=split_dev, do_shuffle=False, name_train="train", name_dev="dev")

    # ------------------------------------

    write_pair_df_tsv(train_df, fn_train_tsv, desc="train")
    write_pair_df_tsv(dev_df, fn_dev_tsv, desc="dev")
    if split_test is not None:
        write_pair_df_tsv(test_df, fn_test_tsv, desc="test")


def write_single_tdt_tsv(root_path, traindev_df, split_test=0.1, split_dev=0.3, do_shuffle=True, random_state=42, binary=True):
    root_path = Path(root_path)
    if not root_path.exists():
        print(f"Create dir: {root_path}")
        root_path.mkdir(parents=True)

    fn_train_tsv = root_path / "train.tsv"
    fn_dev_tsv = root_path / "dev.tsv"
    fn_test_tsv = root_path / "test.tsv"

    # ------------------------------------

    if do_shuffle:
        traindev_df = shuffle(traindev_df, random_state=random_state)

    # ------------------------------------

    if split_test is not None:
        traindev_df, test_df = split_df(traindev_df, ratio=split_test, do_shuffle=False, name_train="traindev", name_dev="test")

    train_df, dev_df = split_df(traindev_df, ratio=split_dev, do_shuffle=False, name_train="train", name_dev="dev")

    # ------------------------------------

    if not binary:
        _write_df_tsv = write_single_5_df_tsv
    else:
        _write_df_tsv = write_single_2_df_tsv

    _write_df_tsv(train_df, fn_train_tsv, desc="train")
    _write_df_tsv(dev_df, fn_dev_tsv, desc="dev")
    if split_test is not None:
        _write_df_tsv(test_df, fn_test_tsv, desc="test")


# ---------------------------------------------------------------------------
# cross eval
# ---------------------------------------------------------------------------


def build_category_business_lookup(map_categories, inv_cat_bids, map_cat_name2id):
    """build lookup of category -> business_ids"""
    lookup_rootcat_bid = dict()

    # --> print_category_tree_with_num_businesses_root(map_categories, inv_cat_bids, map_cat_name2id)
    root_categories = get_root_category_items(map_categories)

    for item in sorted(root_categories, key=lambda x: x["title"]):
        businesses = get_businesses_in_category_branch(inv_cat_bids, item["title"], map_categories, map_cat_name2id)
        businesses = set(businesses)
        lookup_rootcat_bid[(item["title"], item["alias"])] = businesses

    return lookup_rootcat_bid


def filter_category_business_lookup_no_overlap(lookup_rootcat_bid):
    """remove duplicates / overlapping businesses"""
    lookup_rootcat_bid_no_overlap = dict()

    for (title, alias), businesses in lookup_rootcat_bid.items():
        # collect business ids from other categories
        businesses_other = set()
        for (title2, alias2), businesses2 in lookup_rootcat_bid.items():
            if alias2 == alias:
                continue
            businesses_other |= businesses2

        # remove other businesses
        businesses_no_overlap = businesses - businesses_other

        lookup_rootcat_bid_no_overlap[(title, alias)] = businesses_no_overlap
        
    return lookup_rootcat_bid_no_overlap


# ---------------------------------------------------------------------------
# Filter non-overlapping from pairs

def df_add_business_id(df):
    def add_business_id(row):
        bid = row["argument1_id"].split("|", 1)[0]
        row["business_id"] = bid
        return row

    df = df.progress_apply(add_business_id, axis=1)
    return df


def filter_overlapping_businesses(traindev_df, lookup_rootcat_bid_no_overlap):
    """filter no overlapping business"""
    all_business_ids_no_overlap = set()
    for businesses in lookup_rootcat_bid_no_overlap.values():
        all_business_ids_no_overlap |= businesses

    traindev_df = traindev_df[traindev_df.business_id.isin(all_business_ids_no_overlap)]
    return traindev_df


# ---------------------------------------------------------------------------
# cross eval splits


def make_group_split(lookup_rootcat_bid_no_overlap, n=7, seed=42):
    assert n in (4, 7)

    # manual root category group splitting
    cats = list(lookup_rootcat_bid_no_overlap.keys())
    cats.remove(('Bicycles', 'bicycles'))

    # make it repeatable
    set_seed(seed=seed)
    shuffle(cats)  # np.random.shuffle(cats)

    # manually split
    if n == 7:
        groups = cats[0:3], cats[3:6], cats[6:9], cats[9:12], cats[12:15], cats[15:18], cats[18:21]
    elif n == 4:
        groups = cats[0:5], cats[5:10], cats[10:15], cats[15:21]

    groups = [tuple(g) for g in groups]

    return groups


def make_cross_eval_dfs(traindev_df, groups, lookup_rootcat_bid_no_overlap):
    # filter business ids of group
    map_cg_bids = dict()
    for cg_ids in groups:
        cg_businesses = set()
        for ta_id, businesses in lookup_rootcat_bid_no_overlap.items():
            if ta_id in cg_ids:
                cg_businesses |= businesses

        map_cg_bids[cg_ids] = cg_businesses

    # --------------------------------

    # build dataframes for each group
    map_cg_traindev_df = dict()

    for cg_ids in groups:
        businesses = map_cg_bids[cg_ids]
        cg_df = traindev_df[traindev_df.business_id.isin(businesses)]
        map_cg_traindev_df[cg_ids] = cg_df

    # --------------------------------

    # make cross eval split traindev dfs
    map_cg_train_dev_groups = dict()

    for cg_ids, cg_df in map_cg_traindev_df.items():
        train_df = cg_df
        dev_dfs = [
            cg_df_o
            for cg_ids_o, cg_df_o in map_cg_traindev_df.items()
            if cg_ids_o != cg_ids
        ]
        dev_df = pd.concat(dev_dfs)
        map_cg_train_dev_groups[cg_ids] = (train_df, dev_df, dev_dfs)

    # --------------------------------

    return map_cg_train_dev_groups


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Print category trees


def print_category_tree(map_categories):
    root_categories = get_root_category_items(map_categories)
    
    def _print_cat_list_rec(lst_cats, level=0):
        for item in sorted(lst_cats, key=lambda x: x["title"]):
            if level:
                print("  " * level, end="")
            print(f"""{item["title"]} [{item["alias"]}]""", end="")
            if item["children"]:
                print(f""" [#{len(item["children"])} children]""")
            else:
                print()
            
            children = get_children_category_item_list(map_categories, item["alias"])
            _print_cat_list_rec(children, level=level + 1)
            
    _print_cat_list_rec(root_categories, level=0)


def print_category_tree_with_num_businesses(map_categories, inv_cat_bids):
    root_categories = get_root_category_items(map_categories)
    
    def _print_cat_list_rec(lst_cats, level=0):
        for item in sorted(lst_cats, key=lambda x: x["title"]):
            cur_line = " ." * 30
            parts = list()

            if level:
                parts.append("  " * level)
            parts.append(f"""{item["title"]} [{item["alias"]}]""")
            
            str_len = sum(len(part) for part in parts)
            print("".join(part for part in parts), end="")
            print(cur_line[str_len:], end="")
            
            if item["title"] not in inv_cat_bids:
                print(" No businesses associated!")
            else:
                print(f""" {len((inv_cat_bids[item["title"]])):>5d} businesses""")
            
            children = get_children_category_item_list(map_categories, item["alias"])
            _print_cat_list_rec(children, level=level + 1)
            
            if level == 0:
                print()
            
    _print_cat_list_rec(root_categories, level=0)
    

def print_category_tree_with_num_businesses_rec(map_categories, inv_cat_bids, map_cat_name2id):
    root_categories = get_root_category_items(map_categories)
    
    def _print_cat_list_rec(lst_cats, level=0):
        for item in sorted(lst_cats, key=lambda x: x["title"]):
            cur_line = " ." * 30
            parts = list()

            if level:
                parts.append("  " * level)
            parts.append(f"""{item["title"]} [{item["alias"]}]""")
            
            str_len = sum(len(part) for part in parts)
            print("".join(part for part in parts), end="")
            print(cur_line[str_len:], end="")
            
            businesses = get_businesses_in_category_branch(inv_cat_bids, item["title"], map_categories, map_cat_name2id)
            businesses_self = get_businesses_in_category(inv_cat_bids, item["title"])
            if not businesses:
                print(" No businesses associated!")
            else:
                businesses = set(businesses)
                print(f""" {len(businesses):>5d} businesses""", end="")
                if len(businesses) != len(businesses_self):
                    print(f""" (self: {len(businesses_self)})""", end="")
                print()
            
            children = get_children_category_item_list(map_categories, item["alias"])
            _print_cat_list_rec(children, level=level + 1)
            
            if level == 0:
                print()
            
    _print_cat_list_rec(root_categories, level=0)
    
    
def print_category_tree_with_num_businesses_root(map_categories, inv_cat_bids, map_cat_name2id):
    root_categories = get_root_category_items(map_categories)
    
    for item in sorted(root_categories, key=lambda x: x["title"]):
        cur_line = " ." * 25
        parts = [f"""{item["title"]} [{item["alias"]}] """]

        str_len = sum(len(part) for part in parts)
        print("".join(part for part in parts), end="")
        print(cur_line[str_len:], end="")

        businesses = get_businesses_in_category_branch(inv_cat_bids, item["title"], map_categories, map_cat_name2id)
        businesses_self = get_businesses_in_category(inv_cat_bids, item["title"])

        businesses = set(businesses)
        print(f""" {len(businesses):>5d} businesses""", end="")
        if len(businesses) != len(businesses_self):
            print(f""" (self: {len(businesses_self)})""", end="")
        print()
        

def print_category_tree_with_num_businesses_root2(map_categories, inv_cat_bids, map_cat_name2id):
    root_categories = get_root_category_items(map_categories)
    for item in root_categories:
        item["businesses"] = get_businesses_in_category_branch(inv_cat_bids, item["title"], map_categories, map_cat_name2id)
        item["businesses_self"] = get_businesses_in_category(inv_cat_bids, item["title"])
    
    for item in sorted(root_categories, key=lambda x: len(set(x["businesses"]))):
        cur_line = " ." * 25
        parts = [f"""{item["title"]} [{item["alias"]}] """]

        str_len = sum(len(part) for part in parts)
        print("".join(part for part in parts), end="")
        print(cur_line[str_len:], end="")

        businesses = item["businesses"]
        businesses_self = item["businesses_self"]

        businesses = set(businesses)
        print(f""" {len(businesses):>5d} businesses""", end="")
        if len(businesses) != len(businesses_self):
            print(f""" (self: {len(businesses_self)})""", end="")
        print()


# ---------------------------------------------------------------------------


def print_category_tree_with_num_reviews_root(map_categories, inv_cat_bids, map_cat_name2id, df_reviews):
    root_categories = get_root_category_items(map_categories)
    
    for item in sorted(root_categories, key=lambda x: x["title"]):
        cur_line = " ." * 25
        parts = [f"""{item["title"]} [{item["alias"]}] """]

        str_len = sum(len(part) for part in parts)
        print("".join(part for part in parts), end="")
        print(cur_line[str_len:], end="")

        businesses = get_businesses_in_category_branch(inv_cat_bids, item["title"], map_categories, map_cat_name2id)
        # businesses_self = get_businesses_in_category(inv_cat_bids, item["title"])
        
        df_reviews_filtered = filter_by_businesses(df_reviews, businesses)
        num_reviews = df_reviews_filtered.rid.count()

        print(f""" {num_reviews:>8d} reviews""")


# ---------------------------------------------------------------------------
# Make category comparisons


def print_2category_compare(inv_cat_bids, map_categories, map_cat_name2id, cat_name_i, cat_name_j):
    businesses_i = get_businesses_in_category_branch(inv_cat_bids, cat_name_i, map_categories, map_cat_name2id)
    businesses_j = get_businesses_in_category_branch(inv_cat_bids, cat_name_j, map_categories, map_cat_name2id)
    
    cat_name_i += ":"
    cat_name_j += ":"
    width = max(12, len(cat_name_i), len(cat_name_j))

    print(f"""{cat_name_i:<{width}} {len(set(businesses_i)):>5d}""")
    print(f"""{cat_name_j:<{width}} {len(set(businesses_j)):>5d}""")
    print(f"""Both: {"same:":>{width - 6}} {len(set(businesses_i) & set(businesses_j)):>5d}""")
    print(f"""{"total:":>{width}} {len(set(businesses_i) | set(businesses_j)):>5d}""")


def make_NxN_category_businesses_overlap(inv_cat_bids, map_categories, map_cat_name2id):
    root_categories = get_root_category_items(map_categories)
    root_categories = sorted(root_categories, key=lambda x: x["title"])
    root_category_labels = [x["title"] for x in root_categories]
    
    array = list()
    for cname_i in root_category_labels:
        array_line = list()
        for cname_j in root_category_labels:
            businesses_i = get_businesses_in_category_branch(inv_cat_bids, cname_i, map_categories, map_cat_name2id)
            businesses_j = get_businesses_in_category_branch(inv_cat_bids, cname_j, map_categories, map_cat_name2id)
            businesses_i, businesses_j = set(businesses_i), set(businesses_j)
            businesses_ij_union = businesses_i | businesses_j
            businesses_ij_intersect = businesses_i & businesses_j
            num_businesses_ij = len(businesses_ij_intersect)
            #array_line.append(num_businesses_ij)
            array_line.append(len(businesses_ij_intersect) / len(businesses_ij_union))
        array.append(array_line)
    
    df_cm = pd.DataFrame(array, index=list(root_category_labels), columns=list(root_category_labels))
    
    return array, root_category_labels, df_cm


# ---------------------------------------------------------------------------
