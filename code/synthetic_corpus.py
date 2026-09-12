"""
Synthetic corpus generation for TDA-RAG experiments.
Each 'document' is a bag of tokens drawn from topic-specific vocabularies,
with controllable mixing to create clusters, cycles, and transitional chains.
"""
import random
random.seed(42)

VOCAB = {
    "db":       ["index","query","transaction","schema","postgres","sql","join","commit","replica","shard"],
    "cook":     ["simmer","saute","recipe","broth","seasoning","knife","oven","garnish","whisk","marinate"],
    "astro":    ["orbit","nebula","telescope","redshift","galaxy","photon","gravity","exoplanet","spectrum","pulsar"],
    "ml":       ["gradient","tensor","embedding","loss","optimizer","layer","backprop","dataset","epoch","inference"],
}

def make_doc(topics_weights, n_tokens=25, noise=2):
    """topics_weights: dict topic->weight (relative proportion of tokens)."""
    tokens = []
    for topic, w in topics_weights.items():
        k = max(1, int(round(n_tokens * w)))
        tokens += random.choices(VOCAB[topic], k=k)
    # small shared noise vocabulary (generic words) to mimic real corpora
    generic = ["system","data","process","result","model","report","analysis","method"]
    tokens += random.choices(generic, k=noise)
    random.shuffle(tokens)
    return " ".join(tokens)

def pure_cluster(topic, n_docs, n_tokens=25):
    return [make_doc({topic: 1.0}, n_tokens) for _ in range(n_docs)]

def bridge_docs(topic_a, topic_b, n_docs, n_tokens=25):
    return [make_doc({topic_a: 0.5, topic_b: 0.5}, n_tokens) for _ in range(n_docs)]

def chain(topic_a, topic_b, n_steps, n_tokens=25):
    """Progressive blend from pure topic_a to pure topic_b over n_steps documents."""
    docs = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        docs.append(make_doc({topic_a: 1 - t, topic_b: t}, n_tokens))
    return docs
