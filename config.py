# config.py - centralized paths and hyperparameters
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"

NCHLT_ROOT = DATA_ROOT / "NCHLT Text Corpora"
MASAKHA_NER_ROOT = DATA_ROOT / "ner_MasakhaNER 2.0" / "masakhaner2"
BILINGUAL_SEED_LEXICON_ROOT = DATA_ROOT / "Bilingual Seed Lexicons"

EMBEDDINGS_ROOT = PROJECT_ROOT / "embeddings"
RESULTS_ROOT = PROJECT_ROOT / "results"

ENGLISH = "eng"
LANGUAGES = ["zul", "nso", "tsn"]  # isiZulu, Sepedi, Setswana
EMBEDDING_DIM = 100
SEED_LEXICON_SIZE = 5000
HELD_OUT_PAIRS = 1000
CORPUS_FRACTIONS = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05]

# Canonical codes used by the codebase differ from some on-disk NCHLT names.
LANGUAGE_DISPLAY_NAMES = {
    "eng": "English",
    "zul": "isiZulu",
    "nso": "Sepedi",
    "tsn": "Setswana",
}

NCHLT_CORPUS_CLEAN_PATHS = {
    "eng": NCHLT_ROOT / "en" / "corpora" / "1_Corpus_nchlt" / "CORP.NCHLT.eng.CLEAN.1.0.0.txt",
    "zul": NCHLT_ROOT / "zu" / "2.Corpora" / "CORP.NCHLT.zu.CLEAN.2.0.txt",
    "nso": NCHLT_ROOT / "nso" / "2.Corpora" / "CORP.NCHLT.nso.CLEAN.2.0.txt",
    "tsn": NCHLT_ROOT / "tn" / "2.Corpora" / "CORP.NCHLT.tn.CLEAN.2.0.txt",
}

NCHLT_CORPUS_RAW_PATHS = {
    "zul": NCHLT_ROOT / "zu" / "2.Corpora" / "CORP.NCHLT.zu.RAW.2.0.txt",
    "nso": NCHLT_ROOT / "nso" / "2.Corpora" / "CORP.NCHLT.nso.RAW.2.0.txt",
    "tsn": NCHLT_ROOT / "tn" / "2.Corpora" / "CORP.NCHLT.tn.RAW.2.0.txt",
}

NCHLT_LEXICON_FREQ_PATHS = {
    "eng": NCHLT_ROOT / "en" / "lexicons" / "2_Lexica_nchlt" / "Freq.LEX.NCHLT.en.txt",
    "zul": NCHLT_ROOT / "zu" / "3.Lexica" / "FREQ.LEX.NCHLT.zu.txt",
    "nso": NCHLT_ROOT / "nso" / "3.Lexica" / "FREQ.LEX.NCHLT.nso.txt",
    "tsn": NCHLT_ROOT / "tn" / "3.Lexica" / "FREQ.LEX.NCHLT.tn.txt",
}

NCHLT_LEXICON_NE_PATHS = {
    "eng": NCHLT_ROOT / "en" / "lexicons" / "2_Lexica_nchlt" / "NELIST.NCHLT.en.txt",
    "zul": NCHLT_ROOT / "zu" / "3.Lexica" / "NELIST.NCHLT.zu.txt",
    "nso": NCHLT_ROOT / "nso" / "3.Lexica" / "NELIST.NCHLT.nso.txt",
    "tsn": NCHLT_ROOT / "tn" / "3.Lexica" / "NELIST.NCHLT.tn.txt",
}

NCHLT_LEXICON_ALL_PATHS = {
    "eng": NCHLT_ROOT / "en" / "lexicons" / "2_Lexica_nchlt" / "NELIST.NCHLT.all.txt",
    "zul": NCHLT_ROOT / "zu" / "3.Lexica" / "NELIST.NCHLT.all.txt",
    "nso": NCHLT_ROOT / "nso" / "3.Lexica" / "NELIST.NCHLT.all.txt",
    "tsn": NCHLT_ROOT / "tn" / "3.Lexica" / "NELIST.NCHLT.all.txt",
}

BILINGUAL_SEED_LEXICON_PATHS = {
    "zul": BILINGUAL_SEED_LEXICON_ROOT / "zul_en.txt",
    "nso": BILINGUAL_SEED_LEXICON_ROOT / "nso_en.txt",
    "tsn": BILINGUAL_SEED_LEXICON_ROOT / "tsn_en.txt",
}

MASAKHA_NER_DATA_PATHS = {
    "zul": MASAKHA_NER_ROOT / "zul",
    "nso": MASAKHA_NER_ROOT / "nso",
    "tsn": MASAKHA_NER_ROOT / "tsn",
}


def get_display_name(lang):
    return LANGUAGE_DISPLAY_NAMES[lang]


def get_corpus_path(lang):
    return NCHLT_CORPUS_CLEAN_PATHS[lang]


def get_raw_corpus_path(lang):
    return NCHLT_CORPUS_RAW_PATHS.get(lang)


def get_seed_lexicon_path(lang):
    return BILINGUAL_SEED_LEXICON_PATHS.get(lang)


def get_ner_path(lang):
    return MASAKHA_NER_DATA_PATHS.get(lang)


def get_embeddings_path(lang):
    return EMBEDDINGS_ROOT / f"{lang}.bin"


def get_text_embeddings_path(lang):
    return EMBEDDINGS_ROOT / f"{lang}.txt"


def get_aligned_path(lang, method):
    return EMBEDDINGS_ROOT / "aligned" / f"{lang}_{method}.npy"


PATHS = {
    "corpus_clean": NCHLT_CORPUS_CLEAN_PATHS,
    "corpus_raw": NCHLT_CORPUS_RAW_PATHS,
    "lexicon_freq": NCHLT_LEXICON_FREQ_PATHS,
    "lexicon_ne": NCHLT_LEXICON_NE_PATHS,
    "lexicon_all": NCHLT_LEXICON_ALL_PATHS,
    "seed_lexicon": BILINGUAL_SEED_LEXICON_PATHS,
    "ner": MASAKHA_NER_DATA_PATHS,
    "embeddings": str(EMBEDDINGS_ROOT / "{lang}.bin"),
    "aligned": str(EMBEDDINGS_ROOT / "aligned" / "{lang}_{method}.npy"),
}
