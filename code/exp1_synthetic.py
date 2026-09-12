"""
Experiment 1: H0 persistence as multiscale, k-free clustering.
Build 4 disjoint topic clusters (no bridges), embed via TF-IDF+SVD,
compute VR persistent homology, compare H0 structure to ground truth.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
import gudhi
import matplotlib.pyplot as plt
from corpus import pure_cluster

topics = ["db", "cook", "astro", "ml"]
docs, labels = [], []
for t in topics:
    d = pure_cluster(t, n_docs=15)
    docs += d
    labels += [t]*len(d)

vec = TfidfVectorizer()
X_tfidf = vec.fit_transform(docs)
svd = TruncatedSVD(n_components=20, random_state=0)
X = svd.fit_transform(X_tfidf)
X = X / np.linalg.norm(X, axis=1, keepdims=True)  # normalize -> cosine geometry

D = cosine_distances(X)

rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=1.2)
st = rips.create_simplex_tree(max_dimension=2)
diag = st.persistence()

# Extract H0 bars (birth, death)
h0 = sorted([(b, d) for (dim, (b, d)) in diag if dim == 0], key=lambda x: x[1] - x[0], reverse=True)
h1 = sorted([(b, d) for (dim, (b, d)) in diag if dim == 1], key=lambda x: (x[1]-x[0] if x[1]!=float('inf') else 999), reverse=True)

print(f"Number of documents: {len(docs)}, true topics: {len(topics)}")
print(f"Total H0 bars: {len(h0)}")
print("Top 8 longest H0 bars (birth, death, lifetime):")
for b, d in h0[:8]:
    print(f"  birth={b:.4f} death={d:.4f} lifetime={d-b:.4f}")
print(f"H1 bars found: {len(h1)}")

# Gap-based estimate of cluster count: treat the infinite bar as lifetime = max_edge_length,
# then find the largest gap in sorted lifetimes. Number of long bars = estimated cluster count.
max_len = 1.2
lifetimes = sorted([(max_len if d == float('inf') else d) - b for b, d in h0], reverse=True)
gaps = [lifetimes[i] - lifetimes[i+1] for i in range(len(lifetimes)-1)]
cut = int(np.argmax(gaps)) + 1
print(f"Largest gap in H0 lifetimes at rank {cut} -> estimated cluster count = {cut} (ground truth = {len(topics)})")

# Plot barcode
fig, ax = plt.subplots(figsize=(6,4))
for i, (b, d) in enumerate(h0):
    d_plot = 1.2 if d == float('inf') else d
    ax.plot([b, d_plot], [i, i], color='C0', lw=2)
ax.set_xlabel("epsilon (cosine distance)")
ax.set_ylabel("H0 bar index")
ax.set_title("Experiment 1: H0 barcode, 4 disjoint topic clusters (n=60 docs)")
plt.tight_layout()
plt.savefig("/home/claude/tda_rag/fig1_h0_barcode.png", dpi=150)
print("saved fig1")
