"""
Experiment 3: graph-geodesic retrieval vs raw cosine top-k.
Setup: a topic 'db' cluster and a topic 'astro' cluster connected by a
gradual transitional chain (db -> ... -> astro). A query drawn from the
'db' cluster should, semantically, retrieve chain documents in order of
their true blend proximity. We compare:
  (a) ranking by raw cosine distance from the query
  (b) ranking by shortest-path ('geodesic') distance on the epsilon-graph
against ground-truth blend position (0=pure db, 1=pure astro).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from corpus import pure_cluster, chain

random_state = 0
n_chain = 14
docs, blend_pos = [], []

db_core = pure_cluster("db", n_docs=15); docs += db_core; blend_pos += [0.0]*len(db_core)
chain_docs = chain("db", "astro", n_steps=n_chain); docs += chain_docs
blend_pos += list(np.linspace(0,1,n_chain))
astro_core = pure_cluster("astro", n_docs=15); docs += astro_core; blend_pos += [1.0]*len(astro_core)

# Distractor cluster: shares generic vocabulary with db (cosine-close by accident)
# but is NOT topically or graph-connected to the chain -- an unrelated 'cook' cluster.
from corpus import make_doc
distractor = [make_doc({"cook":1.0}, n_tokens=25) for _ in range(15)]
docs += distractor; blend_pos += [None]*len(distractor)  # not on the db-astro axis

vec = TfidfVectorizer()
X_tfidf = vec.fit_transform(docs)
svd = TruncatedSVD(n_components=25, random_state=random_state)
X = svd.fit_transform(X_tfidf)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
D = cosine_distances(X)

query_idx = 0  # a pure db_core document
epsilon = 0.55  # epsilon-graph threshold for building the geodesic graph
adj = np.where(D <= epsilon, D, 0)
graph = csr_matrix(adj)
geo_dist = shortest_path(graph, method='D', directed=False, indices=query_idx)

cos_dist = D[query_idx]

n = len(docs)
idx_chain_astro = list(range(len(db_core), len(db_core)+n_chain)) + \
                  list(range(len(db_core)+n_chain, len(db_core)+n_chain+len(astro_core)))
idx_distractor = list(range(n - len(distractor), n))

k = 10
cos_topk = np.argsort(cos_dist)[:k]
geo_topk = np.argsort(geo_dist)[:k]

def frac_in(idx_list, idx_set):
    idx_set = set(idx_set)
    return sum(1 for i in idx_list if i in idx_set) / len(idx_list)

print(f"Query: pure 'db' document (index {query_idx})")
print(f"Corpus: {len(db_core)} db_core + {n_chain} chain + {len(astro_core)} astro_core + {len(distractor)} distractor(cook) = {n} docs")
print(f"epsilon for geodesic graph = {epsilon}\n")

print(f"Top-{k} by RAW COSINE distance:")
print(f"  fraction from db/chain/astro axis: {frac_in(cos_topk, idx_chain_astro + list(range(len(db_core)))):.2f}")
print(f"  fraction from distractor (cook) cluster: {frac_in(cos_topk, idx_distractor):.2f}")

print(f"\nTop-{k} by GRAPH-GEODESIC distance (epsilon-graph shortest path):")
print(f"  fraction from db/chain/astro axis: {frac_in(geo_topk, idx_chain_astro + list(range(len(db_core)))):.2f}")
print(f"  fraction from distractor (cook) cluster: {frac_in(geo_topk, idx_distractor):.2f}")

# How far into the chain (toward astro) does each method's top-k reach?
chain_positions = {i: blend_pos[i] for i in range(n) if blend_pos[i] is not None}
def max_chain_reach(topk):
    reached = [chain_positions[i] for i in topk if i in chain_positions]
    return max(reached) if reached else 0.0

print(f"\nMax blend-position reached toward astro in top-{k}:")
print(f"  cosine:   {max_chain_reach(cos_topk):.3f}")
print(f"  geodesic: {max_chain_reach(geo_topk):.3f}")

# Spearman correlation between cosine and geodesic rankings (chain+core docs only, finite geo_dist)
finite_mask = np.isfinite(geo_dist)
valid_idx = [i for i in range(n) if finite_mask[i] and i != query_idx]
rho, pval = spearmanr(cos_dist[valid_idx], geo_dist[valid_idx])
n_disconnected = int((~finite_mask).sum())
print(f"\nSpearman rho(cosine, geodesic) over {len(valid_idx)} reachable docs: {rho:.4f} (p={pval:.2e})")
print(f"Documents UNREACHABLE from query at epsilon={epsilon} (geo_dist=inf): {n_disconnected}")
print(f"  of which distractor (cook) cluster: {sum(1 for i in idx_distractor if not finite_mask[i])}/{len(distractor)}")

# Plot: cosine vs geodesic distance, colored by group
fig, ax = plt.subplots(figsize=(6,5))
groups = {
    "db_core": range(0, len(db_core)),
    "chain": range(len(db_core), len(db_core)+n_chain),
    "astro_core": range(len(db_core)+n_chain, len(db_core)+n_chain+len(astro_core)),
    "distractor(cook)": range(n-len(distractor), n),
}
colors = {"db_core":"C0","chain":"C2","astro_core":"C3","distractor(cook)":"C1"}
for g, rng in groups.items():
    rng = [i for i in rng if i != query_idx]
    gx = cos_dist[rng]
    gy = [geo_dist[i] if np.isfinite(geo_dist[i]) else 1.3 for i in rng]
    ax.scatter(gx, gy, label=g, color=colors[g], alpha=0.7, s=30)
ax.axhline(1.25, color='gray', ls=':', lw=1)
ax.text(0.02, 1.27, "unreachable (geo_dist = inf)", fontsize=7, color='gray')
ax.set_xlabel("cosine distance from query")
ax.set_ylabel("graph-geodesic distance from query")
ax.set_title(f"Experiment 3: cosine vs geodesic distance (query = pure db doc)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/home/claude/tda_rag/fig3_geodesic_vs_cosine.png", dpi=150)
print("saved fig3")

# Where do distractor docs fall in the RAW COSINE ranking overall (not just top-k)?
full_cos_rank = np.argsort(np.argsort(cos_dist))  # rank 0 = closest
distractor_cos_ranks = sorted(full_cos_rank[i] for i in idx_distractor)
astro_cos_ranks = sorted(full_cos_rank[i] for i in range(len(db_core)+n_chain, len(db_core)+n_chain+len(astro_core)))
print(f"\nDistractor cosine ranks (out of {n-1}): min={min(distractor_cos_ranks)}, median={int(np.median(distractor_cos_ranks))}, max={max(distractor_cos_ranks)}")
print(f"True astro_core cosine ranks: min={min(astro_cos_ranks)}, median={int(np.median(astro_cos_ranks))}, max={max(astro_cos_ranks)}")
n_distractor_beats_astro = sum(1 for dr in distractor_cos_ranks for ar in astro_cos_ranks if dr < ar)
print(f"Distractor docs cosine-ranked ABOVE (closer than) at least one true astro_core doc: "
      f"{sum(1 for dr in distractor_cos_ranks if dr < max(astro_cos_ranks))}/{len(distractor_cos_ranks)}")
