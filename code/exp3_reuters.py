"""
Final version: k=12 mutual-kNN graph, full metrics, robustness check
across two independent random seeds for document sampling.
"""
import numpy as np
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import wilcoxon
from real_pipeline import single_label_docs, two_label_docs
import random

def sample(fids, n, seed):
    rng = random.Random(seed); fids = list(fids); rng.shuffle(fids)
    return fids[:n]

def mutual_knn_graph(D, k):
    n = D.shape[0]
    nn = np.argsort(D, axis=1)[:, 1:k+1]
    adj = np.zeros((n, n))
    for i in range(n):
        for j in nn[i]:
            if i in nn[j]:
                adj[i, j] = D[i, j]; adj[j, i] = D[i, j]
    return adj

def run(seed, K=12, TOPK=10):
    crude_fids = single_label_docs("crude"); ship_fids = single_label_docs("ship")
    grain_fids = single_label_docs("grain"); earn_fids = single_label_docs("earn")
    cs_bridge = two_label_docs("crude", "ship"); sg_bridge = two_label_docs("ship", "grain")

    used_crude = sample(crude_fids, 40, seed)
    used_cs    = sample(cs_bridge, 8, seed)
    used_ship  = sample(ship_fids, 15, seed)
    used_sg    = sample(sg_bridge, 8, seed)
    used_grain = sample(grain_fids, 20, seed)
    used_earn  = sample(earn_fids, 30, seed)

    blocks = [used_crude, used_cs, used_ship, used_sg, used_grain, used_earn]
    names  = ["crude_core","crude-ship_bridge","ship_core","ship-grain_bridge","grain_core","earn_distractor"]
    sizes  = [len(b) for b in blocks]
    all_fids = sum(blocks, [])
    texts = [reuters.raw(f) for f in all_fids]
    n = len(all_fids)

    idx = {}; cursor = 0
    for nm, sz in zip(names, sizes):
        idx[nm] = list(range(cursor, cursor+sz)); cursor += sz

    vec = TfidfVectorizer(stop_words='english', max_df=0.7, min_df=2)
    Xtf = vec.fit_transform(texts)
    svd = TruncatedSVD(n_components=30, random_state=0)
    X = svd.fit_transform(Xtf)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    D = cosine_distances(X); np.fill_diagonal(D, 0)

    earn_idx = set(idx["earn_distractor"])
    target_idx = set(idx["ship_core"] + idx["grain_core"] + idx["ship-grain_bridge"] + idx["crude-ship_bridge"])
    query_pool = idx["crude_core"]

    adj = mutual_knn_graph(D, K)
    graph = csr_matrix(adj)
    n_edges = int((adj>0).sum()/2)

    cos_contam, geo_contam, cos_recall, geo_recall, cos_sep, geo_sep = [], [], [], [], [], []
    for q in query_pool:
        cos_order = np.argsort(D[q])
        cos_topk = [i for i in cos_order if i != q][:TOPK]
        cos_contam.append(sum(1 for i in cos_topk if i in earn_idx)/TOPK)
        cos_recall.append(sum(1 for i in cos_topk if i in target_idx)/min(TOPK,len(target_idx)))
        rank_cos = {doc:r for r,doc in enumerate(cos_order)}
        cos_sep.append(np.mean([rank_cos[i] for i in earn_idx]) - np.mean([rank_cos[i] for i in target_idx]))

        geo = shortest_path(graph, method='D', directed=False, indices=q)
        geo_order = np.argsort(geo)
        geo_topk = [i for i in geo_order if i != q and np.isfinite(geo[i])][:TOPK]
        geo_contam.append(sum(1 for i in geo_topk if i in earn_idx)/TOPK)
        geo_recall.append(sum(1 for i in geo_topk if i in target_idx)/min(TOPK,len(target_idx)))
        rank_geo = {doc:r for r,doc in enumerate(geo_order)}
        er = [rank_geo[i] for i in earn_idx if np.isfinite(geo[i])]
        tr = [rank_geo[i] for i in target_idx if np.isfinite(geo[i])]
        if er and tr:
            geo_sep.append(np.mean(er) - np.mean(tr))

    cos_contam, geo_contam = np.array(cos_contam), np.array(geo_contam)
    cos_recall, geo_recall = np.array(cos_recall), np.array(geo_recall)
    cos_sep_a = np.array(cos_sep)
    m = len(geo_sep)
    geo_sep_a = np.array(geo_sep)
    stat, p_sep = wilcoxon(cos_sep_a[:m], geo_sep_a)
    stat2, p_contam = wilcoxon(cos_contam, geo_contam)

    print(f"--- seed={seed}, n_docs={n}, mutual-{K}NN edges={n_edges}, queries={len(query_pool)} ---")
    print(f"  contamination: cosine={cos_contam.mean():.3f} geodesic={geo_contam.mean():.3f}  (wilcoxon p={p_contam:.4f})")
    print(f"  recall@{TOPK}:     cosine={cos_recall.mean():.3f} geodesic={geo_recall.mean():.3f}")
    print(f"  rank-separation (earn_rank - target_rank, higher=better): cosine={cos_sep_a.mean():.2f} geodesic={geo_sep_a.mean():.2f}  (wilcoxon p={p_sep:.6f}, n={m})")
    return dict(seed=seed, cos_contam=cos_contam.mean(), geo_contam=geo_contam.mean(),
                cos_recall=cos_recall.mean(), geo_recall=geo_recall.mean(),
                cos_sep=cos_sep_a.mean(), geo_sep=geo_sep_a.mean(), p_sep=p_sep, p_contam=p_contam, n=n, edges=n_edges)

r1 = run(seed=7)
r2 = run(seed=123)
r3 = run(seed=2024)
