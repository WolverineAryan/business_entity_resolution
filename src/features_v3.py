import re
import unicodedata
import jellyfish
from typing import Dict, List, Set, Tuple, Optional
from rapidfuzz import fuzz, distance

# Pre-compiled Regex Patterns
RE_INDIC = re.compile(r'[\u0900-\u0D7F]')
RE_NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9\s]')
RE_SPACES = re.compile(r'\s+')
RE_NUMBERS = re.compile(r'\b\d+\b')
RE_POSTAL_IN = re.compile(r'\b[1-9][0-9]{5}\b')          # Indian 6-digit PIN
RE_POSTAL_US = re.compile(r'\b\d{5}(?:-\d{4})?\b')       # US 5-digit ZIP
RE_POSTAL_FR = re.compile(r'\b(?:0[1-9]|[1-8]\d|9[0-8])\d{3}\b') # French 5-digit code
RE_DOMAIN = re.compile(r'(\.com|\.in|\.org|\.net|\.fr|\.co|\.info|\.biz|@|https?://|www\.)', re.IGNORECASE)

# French legal entities and prefixes
RE_SASU = re.compile(r'\bsociete par actions simplifiee( unipersonnelle)?\b', re.IGNORECASE)
RE_SARL = re.compile(r'\bsociete a responsabilite limitee\b', re.IGNORECASE)
RE_EURL = re.compile(r'\bentreprise unipersonnelle a responsabilite limitee\b', re.IGNORECASE)
RE_SA = re.compile(r'\bsociete anonyme\b', re.IGNORECASE)
RE_SCI = re.compile(r'\bsociete civile immobiliere\b', re.IGNORECASE)
RE_STE = re.compile(r'\b(ste|societe)\b', re.IGNORECASE)
RE_CIE = re.compile(r'\b(cie|compagnie)\b', re.IGNORECASE)
RE_ETS = re.compile(r'\b(ets|etablissement|etablissements)\b', re.IGNORECASE)
RE_GIE = re.compile(r'\bgie\b', re.IGNORECASE)
RE_SELARL = re.compile(r'\bselarl\b', re.IGNORECASE)
RE_SCP = re.compile(r'\bscp\b', re.IGNORECASE)
RE_DR = re.compile(r'\b(dr|docteur)\b', re.IGNORECASE)
RE_ME = re.compile(r'\b(me|maitre)\b', re.IGNORECASE)

# French streets
RE_BD = re.compile(r'\b(boulevard|bld)\b', re.IGNORECASE)
RE_AV = re.compile(r'\b(avenue|ave)\b', re.IGNORECASE)
RE_CH = re.compile(r'\bchemin\b', re.IGNORECASE)
RE_IMP = re.compile(r'\bimpasse\b', re.IGNORECASE)
RE_RTE = re.compile(r'\broute\b', re.IGNORECASE)
RE_ALL = re.compile(r'\ballee\b', re.IGNORECASE)
RE_RUE = re.compile(r'\brue\b', re.IGNORECASE)

# English legal entities
RE_PVT = re.compile(r'\bprivate limited\b', re.IGNORECASE)
RE_LLC = re.compile(r'\blimited liability company\b', re.IGNORECASE)
RE_CORP = re.compile(r'\bcorporation\b', re.IGNORECASE)
RE_INC = re.compile(r'\bincorporated\b', re.IGNORECASE)
RE_LTD = re.compile(r'\blimited\b', re.IGNORECASE)

GENERIC_BRAND_STOPWORDS = {
    'bank', 'shop', 'store', 'cafe', 'auto', 'mart', 'care', 'club', 'food', 'tech',
    'plus', 'life', 'star', 'home', 'city', 'zone', 'gold', 'best', 'fast', 'king',
    'real', 'true', 'east', 'west', 'park', 'mall', 'hair', 'nail', 'spa', 'link',
    'line', 'mini', 'mega', 'super', 'hotel', 'house', 'point', 'group', 'india', 'world'
}

def normalize_text_v3(text: str) -> str:
    """Robust Unicode NFKD accent stripping + entity & street normalization."""
    if not text or not isinstance(text, str):
        return ""
    text_nfkd = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    t = text_nfkd.lower()

    # French corporate forms
    t = RE_SASU.sub('sas', t)
    t = RE_SARL.sub('sarl', t)
    t = RE_EURL.sub('eurl', t)
    t = RE_SA.sub('sa', t)
    t = RE_SCI.sub('sci', t)
    t = RE_STE.sub('ste', t)
    t = RE_CIE.sub('cie', t)
    t = RE_ETS.sub('ets', t)
    t = RE_GIE.sub('gie', t)
    t = RE_SELARL.sub('selarl', t)
    t = RE_SCP.sub('scp', t)
    t = RE_DR.sub('dr', t)
    t = RE_ME.sub('me', t)

    # English corporate forms
    t = RE_PVT.sub('pvt ltd', t)
    t = RE_LLC.sub('llc', t)
    t = RE_CORP.sub('corp', t)
    t = RE_INC.sub('inc', t)
    t = RE_LTD.sub('ltd', t)

    # French streets
    t = RE_BD.sub('bd', t)
    t = RE_AV.sub('av', t)
    t = RE_CH.sub('ch', t)
    t = RE_IMP.sub('imp', t)
    t = RE_RTE.sub('rte', t)
    t = RE_ALL.sub('all', t)
    t = RE_RUE.sub('rue', t)

    return RE_SPACES.sub(' ', RE_NON_ALPHANUM.sub(' ', t)).strip()

STOPWORDS_ACRONYM = {
    'of', 'and', 'de', 'la', 'le', 'the', 'for', 'in', 'at', 'on', 'du', 'des', 'et', 'd', 'l',
    'ste', 'cie', 'ets', 'gie', 'sas', 'sarl', 'pvt', 'ltd', 'inc', 'corp', 'llc'
}

def extract_acronym(clean_name: str) -> str:
    """Extract acronym from words >= 2 letters, filtering common stopwords."""
    words = [w for w in clean_name.split() if w not in STOPWORDS_ACRONYM]
    if len(words) >= 2:
        return "".join(w[0] for w in words if w and w[0].isalnum())
    return ""

def extract_postal_code(addr_raw: str) -> Optional[str]:
    """Extract 5 or 6 digit postal/zip code from raw address."""
    if not addr_raw or not isinstance(addr_raw, str):
        return None
    # 1. Check Indian 6-digit
    m_in = RE_POSTAL_IN.search(addr_raw)
    if m_in:
        return m_in.group(0)
    # 2. Check US 5-digit
    m_us = RE_POSTAL_US.search(addr_raw)
    if m_us:
        return m_us.group(0)[:5]
    # 3. Check French 5-digit
    m_fr = RE_POSTAL_FR.search(addr_raw)
    if m_fr:
        return m_fr.group(0)
    return None

def check_domain_or_substring_match(s1_name: str, cand_name: str) -> bool:
    """High-precision domain and brand substring matching without generic word false positives."""
    has_domain_s1 = bool(RE_DOMAIN.search(s1_name))
    has_domain_c = bool(RE_DOMAIN.search(cand_name))
    s1_clean = RE_DOMAIN.sub('', s1_name).replace(' ', '').lower()
    c_clean = RE_DOMAIN.sub('', cand_name).replace(' ', '').lower()

    # If an actual domain/url suffix was present
    if (has_domain_s1 or has_domain_c) and len(s1_clean) >= 4 and len(c_clean) >= 4:
        if s1_clean == c_clean or s1_clean in c_clean or c_clean in s1_clean:
            return True

    # Substring containment only for distinctive non-generic brand names (>= 7 chars)
    if len(c_clean) >= 7 and len(s1_clean) >= 7:
        if c_clean not in GENERIC_BRAND_STOPWORDS and s1_clean not in GENERIC_BRAND_STOPWORDS:
            if c_clean in s1_clean or s1_clean in c_clean:
                return True
    return False

def get_token_jaccard(t1: Set[str], t2: Set[str]) -> float:
    if not t1 or not t2: return 0.0
    inter = len(t1.intersection(t2))
    union = len(t1.union(t2))
    return inter / union if union > 0 else 0.0

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    if not text or len(text) < n:
        return set([text]) if text else set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))

FEATURE_NAMES_V3 = [
    "name_exact",                 # 0
    "name_lev",                   # 1
    "name_tok_sort",              # 2
    "name_tok_set",               # 3
    "name_partial",               # 4
    "name_jaccard",               # 5
    "name_ngram",                 # 6
    "n_len_diff",                 # 7
    "first_word_match",           # 8
    "domain_match",               # 9
    "acronym_match",              # 10
    "phonetic_soundex_match",     # 11
    "phonetic_metaphone_match",   # 12
    "addr_exact",                 # 13
    "addr_lev",                   # 14
    "addr_tok_sort",              # 15
    "addr_tok_set",               # 16
    "addr_jaccard",               # 17
    "a_len_diff",                 # 18
    "addr_empty",                 # 19
    "num_jaccard",                # 20
    "num_exact",                  # 21
    "num_overlap",                # 22
    "num_contradiction",          # 23
    "postal_exact_match",         # 24
    "postal_contradiction",       # 25
    "interaction_name_addr",      # 26
    "composite_harmonic",         # 27
    "is_s3",                      # 28
    "indic_flag",                 # 29
    "s3_blank_addr_interaction",  # 30
    "name_prefix2_match",         # 31
    "name_addr_length_ratio",     # 32
    "num_digits_s1",              # 33
    "num_digits_cand"             # 34
]

def precompute_entity_v3(name_raw: str, addr_raw: str) -> Tuple:
    n_norm = normalize_text_v3(name_raw)
    a_norm = normalize_text_v3(addr_raw)
    n_tok = n_norm.split()
    n_tok_set = set(n_tok)
    first_w = n_tok[0] if n_tok else ""
    first_two = n_norm[:2] if len(n_norm) >= 2 else n_norm
    acronym = extract_acronym(n_norm)

    # Phonetics
    try:
        soundex = jellyfish.soundex(n_norm) if n_norm else ""
        metaphone = jellyfish.metaphone(n_norm) if n_norm else ""
    except Exception:
        soundex, metaphone = "", ""

    a_tok_set = set(a_norm.split())
    raw_nums = set(RE_NUMBERS.findall(a_norm))
    # Filter out 4-digit years (1900-2099) so they don't corrupt street number comparisons
    nums = {n for n in raw_nums if not (len(n) == 4 and (n.startswith('19') or n.startswith('20')))}
    
    postal = extract_postal_code(str(addr_raw))
    has_indic = bool(RE_INDIC.search(str(name_raw)))
    addr_empty = bool(not a_norm or a_norm == 'nan')

    return (
        n_norm, a_norm, n_tok_set, first_w, first_two, acronym,
        soundex, metaphone, a_tok_set, nums, postal, has_indic, addr_empty
    )

def extract_features_v3(s1_tup: Tuple, s1_ng: Set[str], c_tup: Tuple, is_s3: float) -> List[float]:
    (
        s1_n, s1_a, s1_n_set, s1_fw, s1_p2, s1_acr,
        s1_sdx, s1_mtp, s1_a_set, s1_nums, s1_post, s1_indic, s1_a_empty
    ) = s1_tup

    (
        cn, ca, c_n_set, c_fw, c_p2, c_acr,
        c_sdx, c_mtp, c_a_set, c_nums, c_post, c_indic, c_a_empty
    ) = c_tup

    # 1. Name features
    name_exact = 1.0 if s1_n and s1_n == cn else 0.0
    name_lev = distance.Levenshtein.normalized_similarity(s1_n, cn)
    name_tok_sort = fuzz.token_sort_ratio(s1_n, cn) / 100.0
    name_tok_set = fuzz.token_set_ratio(s1_n, cn) / 100.0
    name_partial = fuzz.partial_ratio(s1_n, cn) / 100.0
    name_jaccard = get_token_jaccard(s1_n_set, c_n_set)
    c_ng = get_char_ngrams(cn, 3)
    name_ngram = get_token_jaccard(s1_ng, c_ng)

    max_nl = max(len(s1_n), len(cn))
    n_len_diff = abs(len(s1_n) - len(cn)) / max_nl if max_nl > 0 else 0.0
    first_word_match = 1.0 if s1_fw and c_fw and s1_fw == c_fw else 0.0
    domain_match = 1.0 if check_domain_or_substring_match(s1_n, cn) else 0.0

    # Acronym & Phonetic features
    acronym_match = 0.0
    if s1_acr and (s1_acr == cn or s1_acr in c_n_set):
        acronym_match = 1.0
    elif c_acr and (c_acr == s1_n or c_acr in s1_n_set):
        acronym_match = 1.0

    phonetic_soundex = 1.0 if s1_sdx and c_sdx and s1_sdx == c_sdx else 0.0
    phonetic_metaphone = 1.0 if s1_mtp and c_mtp and s1_mtp == c_mtp else 0.0

    # 2. Address features
    addr_exact = 1.0 if s1_a and s1_a == ca else 0.0
    addr_lev = distance.Levenshtein.normalized_similarity(s1_a, ca)
    addr_tok_sort = fuzz.token_sort_ratio(s1_a, ca) / 100.0
    addr_tok_set = fuzz.token_set_ratio(s1_a, ca) / 100.0
    addr_jaccard = get_token_jaccard(s1_a_set, c_a_set)

    max_al = max(len(s1_a), len(ca))
    a_len_diff = abs(len(s1_a) - len(ca)) / max_al if max_al > 0 else 0.0
    addr_empty = 1.0 if c_a_empty else 0.0

    # 3. Numeric & Postal features
    num_jaccard = get_token_jaccard(s1_nums, c_nums)
    num_exact = 1.0 if s1_nums and s1_nums == c_nums else 0.0
    num_overlap = float(len(s1_nums.intersection(c_nums)))
    num_contradiction = 1.0 if s1_nums and c_nums and len(s1_nums.intersection(c_nums)) == 0 else 0.0

    postal_exact = 1.0 if s1_post and c_post and s1_post == c_post else 0.0
    postal_contradict = 1.0 if s1_post and c_post and s1_post != c_post else 0.0

    # 4. Interactions & structural features
    interaction = name_tok_set * addr_tok_set
    composite_harmonic = (2 * name_tok_sort * addr_tok_sort) / (name_tok_sort + addr_tok_sort + 1e-6)
    indic_flag = 1.0 if (s1_indic or c_indic) else 0.0
    s3_blank_addr = is_s3 * addr_empty * name_tok_sort
    name_p2_match = 1.0 if s1_p2 and c_p2 and s1_p2 == c_p2 else 0.0
    len_ratio = (len(s1_n) + len(s1_a)) / max(1.0, len(cn) + len(ca))
    num_d_s1 = float(len(s1_nums))
    num_d_c = float(len(c_nums))

    return [
        name_exact, name_lev, name_tok_sort, name_tok_set, name_partial,
        name_jaccard, name_ngram, n_len_diff, first_word_match, domain_match,
        acronym_match, phonetic_soundex, phonetic_metaphone,
        addr_exact, addr_lev, addr_tok_sort, addr_tok_set, addr_jaccard,
        a_len_diff, addr_empty, num_jaccard, num_exact, num_overlap,
        num_contradiction, postal_exact, postal_contradict,
        interaction, composite_harmonic, is_s3, indic_flag,
        s3_blank_addr, name_p2_match, len_ratio, num_d_s1, num_d_c
    ]
