"""High-speed feature extraction module for candidate pair classification.
Computes string, token, numeric, and cross-field similarity metrics on the fly using rapidfuzz.
"""

import re
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from rapidfuzz import fuzz, distance

RE_NUMBERS = re.compile(r'\b\d+\b')


def get_token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extracts character n-grams from text."""
    if not text or len(text) < n:
        return set([text]) if text else set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def extract_pair_features(
    s1_name: str,
    s1_addr: str,
    cand_name: str,
    cand_addr: str,
    blocking_sim: float = 0.0,
    is_s3: float = 0.0,
    **kwargs
) -> List[float]:
    """Extracts 23 predictive similarity features directly on the fly from raw strings."""
    s1_name = str(s1_name) if s1_name else ""
    s1_addr = str(s1_addr) if s1_addr else ""
    cand_name = str(cand_name) if cand_name else ""
    cand_addr = str(cand_addr) if cand_addr else ""

    # 1. Name Features
    name_exact = 1.0 if s1_name and s1_name == cand_name else 0.0
    core_name_exact = name_exact

    name_lev = distance.Levenshtein.normalized_similarity(s1_name, cand_name)
    core_name_lev = name_lev
    name_tok_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    name_tok_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0

    s1_tok_name = set(s1_name.split())
    cand_tok_name = set(cand_name.split())
    name_jaccard = get_token_jaccard(s1_tok_name, cand_tok_name)
    name_ngram_jaccard = get_token_jaccard(get_char_ngrams(s1_name, 3), get_char_ngrams(cand_name, 3))

    max_len_name = max(len(s1_name), len(cand_name))
    name_len_diff = abs(len(s1_name) - len(cand_name)) / max_len_name if max_len_name > 0 else 0.0

    # 2. Address Features
    addr_exact = 1.0 if s1_addr and s1_addr == cand_addr else 0.0
    addr_lev = distance.Levenshtein.normalized_similarity(s1_addr, cand_addr)
    addr_tok_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
    addr_tok_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0

    s1_tok_addr = set(s1_addr.split())
    cand_tok_addr = set(cand_addr.split())
    addr_jaccard = get_token_jaccard(s1_tok_addr, cand_tok_addr)

    max_len_addr = max(len(s1_addr), len(cand_addr))
    addr_len_diff = abs(len(s1_addr) - len(cand_addr)) / max_len_addr if max_len_addr > 0 else 0.0

    # 3. Numeric & Postal Features
    s1_nums = set(RE_NUMBERS.findall(s1_addr))
    cand_nums = set(RE_NUMBERS.findall(cand_addr))
    num_jaccard = get_token_jaccard(s1_nums, cand_nums)
    num_overlap_count = float(len(s1_nums.intersection(cand_nums)))
    num_exact_match = 1.0 if s1_nums and s1_nums == cand_nums else 0.0

    # 4. Composite & Interaction Features
    interaction_score = name_tok_set * addr_tok_set
    composite_harmonic = (2 * name_tok_sort * addr_tok_sort) / (name_tok_sort + addr_tok_sort + 1e-6)

    return [
        name_exact,
        core_name_exact,
        name_lev,
        core_name_lev,
        name_tok_sort,
        name_tok_set,
        name_partial,
        name_jaccard,
        name_ngram_jaccard,
        name_len_diff,
        addr_exact,
        addr_lev,
        addr_tok_sort,
        addr_tok_set,
        addr_jaccard,
        addr_len_diff,
        num_jaccard,
        num_overlap_count,
        num_exact_match,
        blocking_sim,
        interaction_score,
        composite_harmonic,
        is_s3
    ]


FEATURE_NAMES = [
    "name_exact",
    "core_name_exact",
    "name_lev",
    "core_name_lev",
    "name_tok_sort",
    "name_tok_set",
    "name_partial",
    "name_jaccard",
    "name_ngram_jaccard",
    "name_len_diff",
    "addr_exact",
    "addr_lev",
    "addr_tok_sort",
    "addr_tok_set",
    "addr_jaccard",
    "addr_len_diff",
    "num_jaccard",
    "num_overlap_count",
    "num_exact_match",
    "blocking_sim",
    "interaction_score",
    "composite_harmonic",
    "is_s3"
]
