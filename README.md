# RAG as a Metric Space: Persistent Homology Diagnostics for RAG Corpora

Code and paper for *Retrieval-Augmented Generation as a Metric Space: Persistent Homology Diagnostics for RAG Corpora*.

Retrieval-augmented generation reduces, operationally, to a distance computation over an embedded corpus. This repository develops a mapping from three corpus-level RAG diagnostics — topic granularity, coverage gaps, and lexical false positives in retrieval — to persistent homology invariants, and evaluates it on a controlled synthetic corpus and on the Reuters-21578 news corpus.

## Contents

```
paper/      LaTeX source and compiled PDF
figures/    All figures used in the paper (conceptual and experimental)
code/       Experiment scripts, organized by corpus and experiment
```

## Code

| File | Description |
|---|---|
| `code/synthetic_corpus.py` | Synthetic corpus generator (disjoint-vocabulary topics, bridge documents, blend chains) |
| `code/exp1_synthetic.py` | Topic-count recovery via $H_0$, synthetic corpus |
| `code/exp2_synthetic.py` | Coverage-gap detection via $H_1$, synthetic ring corpus |
| `code/exp3_synthetic.py` | Graph-geodesic reranking, synthetic chain + distractor |
| `code/reuters_pipeline.py` | Reuters-21578 loading, GloVe mean-pooling, sampling utilities |
| `code/exp1_reuters.py` | Topic-count recovery via $H_0$, Reuters-21578 (pooled embeddings) |
| `code/exp2_reuters.py` | Coverage-gap detection via $H_1$, Reuters-21578 category-cooccurrence ring |
| `code/exp3_reuters.py` | Mutual-$k$NN graph-geodesic reranking, Reuters-21578 |
| `code/make_prelim_figs.py` | Conceptual Vietoris–Rips filtration and barcode illustration |
| `code/make_knn_schematic.py` | Mutual-$k$NN vs. $\epsilon$-ball graph schematic |

## Requirements

```
gudhi
scikit-learn
scipy
numpy
matplotlib
gensim
nltk
```

`nltk` requires the `reuters`, `punkt`, `punkt_tab`, and `stopwords` corpora (`nltk.download(...)`); `gensim` will download `glove-wiki-gigaword-50` on first use (~66 MB).

## Reproducing the paper

Each `exp*.py` script is self-contained and can be run directly; random seeds are fixed in-script. Figures are written to `figures/` by the corresponding `make_*` or `exp*` script. The paper (`paper/paper.tex`) references these figures directly.

## Citation

If you use this code, please cite the accompanying paper (see `paper/paper.tex` for the full reference once assigned an arXiv identifier).
