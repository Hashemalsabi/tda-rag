"""
Conceptual illustrations for the Preliminaries section: a small point
cloud (4 clusters arranged in a ring, connected pairwise by bridge
points but with no diagonal chords) is used to illustrate the
Vietoris-Rips filtration growing with epsilon, and the corresponding
persistence barcode. This is a real (if small) computed example, not
a hand-drawn schematic: all edges, triangles, and bars are computed
from actual pairwise distances via GUDHI.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import gudhi
from itertools import combinations

rng = np.random.RandomState(3)

centers = {
    "A": (0, 0), "B": (6, 0), "C": (6, 6), "D": (0, 6),
}
pts = []
labels = []
for name, (cx, cy) in centers.items():
    for _ in range(4):
        pts.append((cx + rng.normal(0, 0.35), cy + rng.normal(0, 0.35)))
        labels.append(name)

bridges = {
    "AB": (3, -0.4), "BC": (6.4, 3), "CD": (3, 6.4), "DA": (-0.4, 3),
}
for name, (x, y) in bridges.items():
    pts.append((x, y)); labels.append(name)

pts = np.array(pts)
n = len(pts)
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        D[i, j] = np.linalg.norm(pts[i] - pts[j])

max_edge = D.max()
rips = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_edge)
st = rips.create_simplex_tree(max_dimension=2)
diag = st.persistence()

h0 = sorted([(b, d) for (dim, (b, d)) in diag if dim == 0], key=lambda x: x[0])
h1 = sorted([(b, d) for (dim, (b, d)) in diag if dim == 1], key=lambda x: x[0])
print("H0 bars:", [(round(b,2), round(d,2) if d!=float('inf') else 'inf') for b,d in h0])
print("H1 bars:", [(round(b,2), round(d,2) if d!=float('inf') else 'inf') for b,d in h1])

# pick 4 illustrative epsilon snapshots
h1_birth = h1[0][0] if h1 else None
h1_death = h1[0][1] if h1 else None
eps_panels = [0.9, h1_birth * 0.9, (h1_birth + h1_death) / 2, h1_death * 1.05]
print("panel epsilons:", eps_panels)

def draw_complex(ax, eps, title):
    # edges
    for i, j in combinations(range(n), 2):
        if D[i, j] <= eps:
            ax.plot([pts[i,0], pts[j,0]], [pts[i,1], pts[j,1]], color='gray', lw=0.8, zorder=1)
    # filled triangles (2-simplices): all 3 pairwise distances <= eps
    for i, j, k in combinations(range(n), 3):
        if D[i,j] <= eps and D[j,k] <= eps and D[i,k] <= eps:
            tri = Polygon([pts[i], pts[j], pts[k]], closed=True, color='C0', alpha=0.15, zorder=0)
            ax.add_patch(tri)
    ax.scatter(pts[:,0], pts[:,1], color='C0', s=35, zorder=2, edgecolor='white', linewidth=0.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-2, 8); ax.set_ylim(-2, 8)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
panel_titles = [
    r"$\epsilon={:.2f}$: isolated clusters".format(eps_panels[0]),
    r"$\epsilon={:.2f}$: clusters connected,".format(eps_panels[1]) + "\nring not yet closed",
    r"$\epsilon={:.2f}$: ring formed,".format(eps_panels[2]) + "\n" + r"$H_1$ loop alive",
    r"$\epsilon={:.2f}$: diagonal appears,".format(eps_panels[3]) + "\nloop filled in",
]
for ax, eps, title in zip(axes, eps_panels, panel_titles):
    draw_complex(ax, eps, title)
plt.tight_layout()
plt.savefig("/home/claude/tda_rag/prelim_filtration.png", dpi=150)
print("saved prelim_filtration.png")

# barcode figure
fig, ax = plt.subplots(figsize=(6.5, 4))
y = 0
for b, d in h0:
    d_plot = max_edge if d == float('inf') else d
    ax.plot([b, d_plot], [y, y], color='C0', lw=2.2)
    y += 1
y += 1
for b, d in h1:
    d_plot = max_edge if d == float('inf') else d
    ax.plot([b, d_plot], [y, y], color='C1', lw=2.2)
    y += 1
for eps in eps_panels:
    ax.axvline(eps, color='gray', ls=':', lw=0.8)
ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel("bar index")
ax.text(0.02, y+0.5, r"$H_0$ (blue), $H_1$ (orange)", fontsize=9, transform=ax.get_yaxis_transform())
plt.tight_layout()
plt.savefig("/home/claude/tda_rag/prelim_barcode.png", dpi=150)
print("saved prelim_barcode.png")
