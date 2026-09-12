"""
Final Experiment 1: topic-level persistence on Reuters, using
mean-pooled TF-IDF+SVD embeddings aggregated over groups of articles
(the natural granularity for a corpus-level topic diagnostic), and the
principled connected-component count beta_0(epsilon) = 1 + #{finite H0
bars alive at epsilon}, applied identically to the synthetic and real
experiments.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
import gudhi
import matplotlib.pyplot as plt
from nltk.corpus import reuters
from real_pipeline import single_label_docs
import random

CATS = ["coffee", "gold", "ship", "grain"]
GROUP = 8
N_PSEUDO = 5
SEED = 42
SVD_DIM = 25

texts_all, cursor_map = [], {}
cursor = 0
for cat in CATS:
    fids = single_label_docs(cat)
    random.Random(SEED).shuffle(fids)
    n_this = min(N_PSEUDO*GROUP, len(fids))
    chosen = fids[:n_this]
    for f in chosen:
        texts_all.append(reuters.raw(f))
    cursor_map[cat] = (cursor, cursor+n_this)
    cursor += n_this

vec = TfidfVectorizer(stop_words='english', max_df=0.6, min_df=2, sublinear_tf=True)
Xtf = vec.fit_transform(texts_all)
svd = TruncatedSVD(n_components=SVD_DIM, random_state=0)
X = svd.fit_transform(Xtf)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

pseudo, pseudo_labels = [], []
for cat in CATS:
    start, end = cursor_map[cat]
    ids = list(range(start, end))
    n_this = end - start
    for i in range(0, n_this - n_this % GROUP, GROUP):
        grp = ids[i:i+GROUP]
        v = X[grp].mean(axis=0); v = v/(np.linalg.norm(v)+1e-12)
        pseudo.append(v); pseudo_labels.append(cat)

Xp = np.array(pseudo)
print(f"n topic-level embeddings: {Xp.shape[0]} ({N_PSEUDO} per category, "
      f"each pooled over {GROUP} articles => {Xp.shape[0]*GROUP} articles total)")

D = cosine_distances(Xp)
max_edge = float(D.max())
rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_edge)
st = rips.create_simplex_tree(max_dimension=1)
diag = st.persistence()
h0 = sorted([(b,d) for (dim,(b,d)) in diag if dim==0],
            key=lambda x: (x[1]-x[0]) if x[1]!=float('inf') else float('inf'), reverse=True)
finite = sorted([d-b for b,d in h0 if d!=float('inf')], reverse=True)
gaps = [finite[i]-finite[i+1] for i in range(len(finite)-1)]
cut = int(np.argmax(gaps))+1
est_k = cut+1
print(f"top finite lifetimes: {[round(x,3) for x in finite[:6]]}")
print(f"largest gap after {cut} significant finite bars -> beta_0 estimate = 1 + {cut} = {est_k} (true={len(CATS)})")

fig, ax = plt.subplots(figsize=(6,4))
for i, (b, d) in enumerate(h0):
    d_plot = max_edge if d == float('inf') else d
    ax.plot([b, d_plot], [i, i], color='C0', lw=2)
ax.axvline(finite[cut-1], color='gray', ls=':', lw=1)
ax.set_xlabel(r"$\epsilon$ (cosine distance)")
ax.set_ylabel(r"$H_0$ bar index")
ax.set_title(f"$H_0$ barcode, topic-level embeddings\n({Xp.shape[0]} pooled samples, 4 Reuters topics)")
plt.tight_layout()
plt.savefig("/home/claude/tda_rag/fig1_final.png", dpi=150)
print("saved fig1_final.png")
