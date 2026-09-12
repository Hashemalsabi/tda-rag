"""
Experiment 2: H1 as a coverage-gap detector.
Arrange 4 topic clusters in a ring: db-cook-astro-ml-db, with pairwise
bridge documents only between ring-adjacent topics (no direct db-astro
or cook-ml bridge). This should produce a persistent H1 loop: the ring
is connected pairwise but nothing 'fills it in'.
Then add a direct db-astro bridge (a chord) and show the loop dies.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
import gudhi
import matplotlib.pyplot as plt
from corpus import pure_cluster, bridge_docs

ring = ["db", "cook", "astro", "ml"]  # ring order: db-cook, cook-astro, astro-ml, ml-db

def build(with_chord=False):
    docs, labels = [], []
    for t in ring:
        d = pure_cluster(t, n_docs=10)
        docs += d; labels += [f"{t}_core"]*len(d)
    ring_pairs = [("db","cook"), ("cook","astro"), ("astro","ml"), ("ml","db")]
    for a,b in ring_pairs:
        d = bridge_docs(a,b,n_docs=4)
        docs += d; labels += [f"{a}-{b}_bridge"]*len(d)
    if with_chord:
        d = bridge_docs("db","astro", n_docs=4)
        docs += d; labels += ["db-astro_chord"]*len(d)
    return docs, labels

def run(docs, max_edge=1.2):
    vec = TfidfVectorizer()
    X_tfidf = vec.fit_transform(docs)
    svd = TruncatedSVD(n_components=20, random_state=0)
    X = svd.fit_transform(X_tfidf)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    D = cosine_distances(X)
    rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_edge)
    st = rips.create_simplex_tree(max_dimension=2)
    diag = st.persistence()
    h1 = [(b,d) for (dim,(b,d)) in diag if dim==1]
    return h1, diag

print("=== Ring WITHOUT chord (db-astro not directly bridged) ===")
docs_no_chord, labels_no_chord = build(with_chord=False)
h1_a, diag_a = run(docs_no_chord)
h1_a_sorted = sorted(h1_a, key=lambda x: (x[1]-x[0] if x[1]!=float('inf') else 999), reverse=True)
print(f"n_docs={len(docs_no_chord)}, H1 bars={len(h1_a)}")
for b,d in h1_a_sorted[:5]:
    lt = (d-b) if d!=float('inf') else float('inf')
    print(f"  birth={b:.4f} death={d:.4f} lifetime={lt}")

print("\n=== Ring WITH chord (db-astro bridge added) ===")
docs_chord, labels_chord = build(with_chord=True)
h1_b, diag_b = run(docs_chord)
h1_b_sorted = sorted(h1_b, key=lambda x: (x[1]-x[0] if x[1]!=float('inf') else 999), reverse=True)
print(f"n_docs={len(docs_chord)}, H1 bars={len(h1_b)}")
for b,d in h1_b_sorted[:5]:
    lt = (d-b) if d!=float('inf') else float('inf')
    print(f"  birth={b:.4f} death={d:.4f} lifetime={lt}")

max_lt_a = max([(d-b) for b,d in h1_a if d!=float('inf')], default=0)
max_lt_b = max([(d-b) for b,d in h1_b if d!=float('inf')], default=0)
print(f"\nMax finite H1 lifetime WITHOUT chord: {max_lt_a:.4f}")
print(f"Max finite H1 lifetime WITH chord:    {max_lt_b:.4f}")

# Plot both barcodes side by side
fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
for ax, h1, title in [(axes[0], h1_a_sorted[:10], "No direct db-astro bridge\n(ring topology, H1 loop)"),
                       (axes[1], h1_b_sorted[:10], "With direct db-astro bridge\n(chord fills the loop)")]:
    for i,(b,d) in enumerate(h1):
        d_plot = 1.2 if d==float('inf') else d
        ax.plot([b,d_plot],[i,i],color='C1',lw=3)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("epsilon (cosine distance)")
axes[0].set_ylabel("H1 bar index")
plt.tight_layout()
plt.savefig("/home/claude/tda_rag/fig2_h1_gap.png", dpi=150)
print("saved fig2")
