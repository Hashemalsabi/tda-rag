"""
Shared pipeline: real Reuters-21578 documents + real pretrained GloVe
(glove-wiki-gigaword-50) embeddings via mean-pooled word vectors.
"""
import re
import random
import numpy as np
from nltk.corpus import reuters, stopwords
from gensim.models import KeyedVectors

random.seed(42)

STOP = set(stopwords.words('english'))
GLOVE = KeyedVectors.load('/home/claude/tda_rag/glove50.kv')

def tokenize(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if w not in STOP and w in GLOVE]

def doc_embedding(fid):
    text = reuters.raw(fid)
    toks = tokenize(text)
    if len(toks) == 0:
        return None
    vecs = GLOVE[toks]
    v = vecs.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None

def single_label_docs(cat, min_words=30, exclude=None):
    out = []
    exclude = exclude or set()
    for fid in reuters.fileids(cat):
        if fid in exclude:
            continue
        if reuters.categories(fid) == [cat]:
            words = re.findall(r"[a-zA-Z]+", reuters.raw(fid))
            if len(words) >= min_words:
                out.append(fid)
    return out

def two_label_docs(cat_a, cat_b, min_words=20):
    out = []
    target = tuple(sorted([cat_a, cat_b]))
    for fid in reuters.fileids():
        cats = reuters.categories(fid)
        if len(cats) == 2 and tuple(sorted(cats)) == target:
            words = re.findall(r"[a-zA-Z]+", reuters.raw(fid))
            if len(words) >= min_words:
                out.append(fid)
    return out

def sample_embeddings(fids, n=None, seed=42):
    rng = random.Random(seed)
    fids = list(fids)
    rng.shuffle(fids)
    if n is not None:
        fids = fids[:n]
    embs, used = [], []
    for fid in fids:
        e = doc_embedding(fid)
        if e is not None:
            embs.append(e)
            used.append(fid)
    return np.array(embs), used
