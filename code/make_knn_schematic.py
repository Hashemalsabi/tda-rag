"""
Illustrates why a global epsilon-ball graph can be misled by a single
intermediate point ('hub') that is moderately close to two otherwise
unrelated clusters, while a mutual k-nearest-neighbor graph is not:
the hub is not reciprocally among either cluster's nearest neighbors.
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

rng = np.random.RandomState(7)
A = rng.normal(loc=(0,0), scale=0.5, size=(6,2))
B = rng.normal(loc=(8,0), scale=0.5, size=(6,2))
hub = np.array([[4.0, 0.4]])
pts = np.vstack([A, B, hub])
n = len(pts)
groups = ["A"]*6 + ["B"]*6 + ["hub"]
colors = {"A":"C0", "B":"C2", "hub":"C3"}

D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        D[i,j] = np.linalg.norm(pts[i]-pts[j])

def knn(i, k):
    order = np.argsort(D[i])
    return [j for j in order if j != i][:k]

k = 3
mutual_edges = []
for i, j in combinations(range(n), 2):
    if j in knn(i,k) and i in knn(j,k):
        mutual_edges.append((i,j))

eps = 4.3
eps_edges = [(i,j) for i,j in combinations(range(n),2) if D[i,j] <= eps]

fig, axes = plt.subplots(1, 2, figsize=(10,4.2))

ax = axes[0]
for i,j in eps_edges:
    ax.plot([pts[i,0],pts[j,0]],[pts[i,1],pts[j,1]], color='gray', lw=1, zorder=1)
for g in ["A","B","hub"]:
    idx = [i for i in range(n) if groups[i]==g]
    ax.scatter(pts[idx,0], pts[idx,1], color=colors[g], s=60, zorder=2, label=g, edgecolor='white')
ax.set_title(rf"$\epsilon$-ball graph ($\epsilon={eps}$)" + "\nhub bridges A and B")
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
ax.legend(fontsize=8, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)

ax = axes[1]
for i,j in mutual_edges:
    ax.plot([pts[i,0],pts[j,0]],[pts[i,1],pts[j,1]], color='gray', lw=1, zorder=1)
for g in ["A","B","hub"]:
    idx = [i for i in range(n) if groups[i]==g]
    ax.scatter(pts[idx,0], pts[idx,1], color=colors[g], s=60, zorder=2, label=g, edgecolor='white')
ax.set_title(f"mutual {k}-NN graph\nhub not reciprocated, A and B separate")
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("/home/claude/tda_rag/method_knn_schematic.png", dpi=150)
print("saved method_knn_schematic.png")
print("eps-ball edges:", len(eps_edges), "mutual-knn edges:", len(mutual_edges))
print("hub eps-ball degree:", sum(1 for i,j in eps_edges if i==n-1 or j==n-1))
print("hub mutual-knn degree:", sum(1 for i,j in mutual_edges if i==n-1 or j==n-1))
