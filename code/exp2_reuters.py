"""
Experiment 2 (real data): a genuine, naturally occurring coverage-gap
ring in Reuters-21578. Real multi-label articles create real bridges:
crude-ship (75 articles), ship-grain (22), grain-trade (5), and a very
thin trade-crude link (only 1 article in the whole 1987 corpus).
Real chords (direct crude-grain or ship-trade articles) do not exist
in the corpus at all (0 articles) -- this is a naturally occurring gap,
not a constructed one.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import gudhi
import matplotlib.pyplot as plt
from real_pipeline import single_label_docs, two_label_docs, sample_embeddings

RING = ["crude", "ship", "grain", "trade"]
RING_EDGES = [("crude","ship"), ("ship","grain"), ("grain","trade"), ("trade","crude")]

def build(n_core=12, n_bridge_cap=6):
    embs, labels = [], []
    for cat in RING:
        fids = single_label_docs(cat)
        X, used = sample_embeddings(fids, n=n_core, seed=1)
        embs.append(X); labels += [f"{cat}_core"]*len(used)
    edge_counts = {}
    for a, b in RING_EDGES:
        fids = two_label_docs(a, b)
        n = min(n_bridge_cap, len(fids))
        X, used = sample_embeddings(fids, n=n, seed=2)
        edge_counts[(a,b)] = len(used)
        if len(used) > 0:
            embs.append(X); labels += [f"{a}-{b}_bridge"]*len(used)
    return np.vstack(embs), labels, edge_counts

X, labels, edge_counts = build()
print(f"n_docs={X.shape[0]}")
print("Real bridge article counts used:", edge_counts)
print("(full corpus counts: crude-ship=75, ship-grain=22, grain-trade=5, trade-crude=1;"
      " crude-grain=0, ship-trade=0 -- chords absent from the real corpus)")

D = cosine_distances(X)
max_edge = float(D.max())
rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_edge)
st = rips.create_simplex_tree(max_dimension=2)
diag = st.persistence()
h1 = [(b,d) for (dim,(b,d)) in diag if dim==1]
h1_sorted = sorted(h1, key=lambda x: (x[1]-x[0] if x[1]!=float('inf') else 999), reverse=True)
print(f"H1 bars: {len(h1)}")
for b,d in h1_sorted[:5]:
    lt = (d-b) if d!=float('inf') else float('inf')
    print(f"  birth={b:.4f} death={d:.4f} lifetime={lt}")

max_lt = max([(d-b) for b,d in h1 if d!=float('inf')], default=0)
print(f"Max finite H1 lifetime (real ring, no chord): {max_lt:.4f}")

# Now add a synthetic-free but REAL closing test: what if we add real crude-ship AND real
# ship-grain bridges' embeddings averaged as a stand-in "near-chord" is not honest -- instead
# we report the natural result as-is, since no real crude-grain or ship-trade article exists to add.

fig, ax = plt.subplots(figsize=(6,4))
for i,(b,d) in enumerate(h1_sorted[:10]):
    d_plot = max_edge if d==float('inf') else d
    ax.plot([b,d],[i,i],color='C1',lw=3)
ax.set_xlabel("epsilon (cosine distance)")
ax.set_ylabel("H1 bar index")
ax.set_title("Real data: H1 barcode, Reuters crude-ship-grain-trade ring\n(no real crude-grain or ship-trade article exists in corpus)")
plt.tight_layout()
plt.savefig("/home/claude/tda_rag/real_fig2_h1.png", dpi=150)
print("saved real_fig2")
