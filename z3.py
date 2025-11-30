import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import ndcg_score
from colorama import Fore, init

init(autoreset=True)

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    DATASET_ID = "TIGER-Lab/M-BEIR"
    CONFIG_NAME = "query"  
    MODEL_NAME = 'intfloat/multilingual-e5-large'
    
    SIMHASH_BITS = 64
    HAMMING_THRESHOLD = 16   # loosen filter
    NUM_QUERIES = 50
    DISTRACTORS = 150
    TOP_K = 10


# ==========================================
# 2. SIMHASH
# ==========================================
def simhash_vector(vec, n_bits=64, seed=42):
    rng = np.random.default_rng(seed)
    dims = vec.shape[0]
    hyperplanes = rng.standard_normal(size=(n_bits, dims))
    proj = hyperplanes @ vec
    bits = (proj > 0).astype(np.uint8)
    value = 0
    for i, b in enumerate(bits):
        value |= (b << i)
    return value


def hamming(a, b):
    return bin(a ^ b).count("1")


# ==========================================
# 3. METRICS
# ==========================================
def recall_at_k(gold_id, retrieved_ids, k):
    return 1.0 if gold_id in retrieved_ids[:k] else 0.0


def mrr_at_k(gold_id, retrieved_ids, k):
    for rank, doc in enumerate(retrieved_ids[:k], 1):
        if doc == gold_id:
            return 1.0 / rank
    return 0.0


def precision_at_k(gold_id, retrieved_ids, k):
    return 1.0 / k if gold_id in retrieved_ids[:k] else 0.0


def ndcg(gold_id, retrieved_ids, k):
    relevances = np.zeros(k)
    scores = np.zeros(k)
    if gold_id in retrieved_ids[:k]:
        idx = retrieved_ids.index(gold_id)
        relevances[0] = 1
        scores[0] = 1 / np.log2(idx + 2)
    return ndcg_score([relevances], [scores], k=k)


# ==========================================
# 4. NMHG LOCAL SEARCH
# ==========================================
def nmhg_search(q_vec, cand_vecs, cand_ids, q_hash, hthr=16):
    survivors = []
    start_prune = time.time()
    for i, vec in enumerate(cand_vecs):
        chash = simhash_vector(vec, n_bits=Config.SIMHASH_BITS, seed=999+i)
        if hamming(q_hash, chash) <= hthr:
            survivors.append(i)
    prune_time = time.time() - start_prune

    if not survivors:
        survivors = list(range(min(5, len(cand_vecs))))

    survivor_vecs = cand_vecs[survivors]
    survivor_ids = [cand_ids[i] for i in survivors]

    start_dense = time.time()
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    survivor_norms = survivor_vecs / (np.linalg.norm(survivor_vecs, axis=1, keepdims=True) + 1e-9)
    scores = np.dot(survivor_norms, q_norm)
    ranked = np.argsort(-scores)[:Config.TOP_K]
    dense_time = time.time() - start_dense

    return [survivor_ids[i] for i in ranked], prune_time, dense_time, len(survivors)


def dense_baseline(q_vec, cand_vecs, cand_ids):
    start = time.time()
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    mat = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)
    scores = np.dot(mat, q_norm)
    idxs = np.argsort(-scores)[:Config.TOP_K]
    return [cand_ids[i] for i in idxs], time.time() - start


# ==========================================
# 5. MAIN BENCHMARK
# ==========================================
def run_benchmark():
    print(f"{Fore.CYAN}Loading M-BEIR Stream...")
    ds = load_dataset(Config.DATASET_ID, Config.CONFIG_NAME, split="train", streaming=True)
    it = iter(ds)

    print(f"{Fore.CYAN}Loading encoder ({Config.MODEL_NAME})...")
    model = SentenceTransformer(Config.MODEL_NAME)

    recall_nmhg, recall_dense = [], []
    mrr_nmhg, mrr_dense = [], []
    prec_nmhg, prec_dense = [], []
    ndcg_nmhg, ndcg_dense = [], []

    prune_ratios = []
    prune_latencies = []
    dense_latencies = []
    nmhg_latencies = []
    fn_rates = []

    for q in range(Config.NUM_QUERIES):
        print(f"{Fore.YELLOW}Query {q+1}/{Config.NUM_QUERIES}")
        row = next(it)

        qtext = row["query_txt"]
        pos = row["pos_cand_list"][0]
        negs = row.get("neg_cand_list", [])

        while len(negs) < Config.DISTRACTORS:
            negs.append(f"dummy negative text {len(negs)}")

        cand_texts = [pos] + negs[:Config.DISTRACTORS]
        cand_ids = ["gold"] + [f"n{k}" for k in range(len(negs))]

        q_vec = model.encode(f"query: {qtext}", normalize_embeddings=True)
        cand_vecs = model.encode([f"passage: {x}" for x in cand_texts], normalize_embeddings=True)

        # DENSE baseline
        dense_ids, dense_t = dense_baseline(q_vec, cand_vecs, cand_ids)

        # NMHG
        q_hash = simhash_vector(q_vec, n_bits=Config.SIMHASH_BITS)
        nmhg_ids, prune_t, nmhg_dense_t, survivors = nmhg_search(
            q_vec, cand_vecs, cand_ids, q_hash, hthr=Config.HAMMING_THRESHOLD
        )

        nmhg_total_t = prune_t + nmhg_dense_t
        nmhg_latencies.append(nmhg_total_t)
        dense_latencies.append(dense_t)
        prune_latencies.append(prune_t)

        # pruning rate
        prune_ratio = survivors / len(cand_ids)
        prune_ratios.append(prune_ratio)

        # false negative rate
        fn_rates.append(0.0 if "gold" in nmhg_ids else 1.0)

        # METRICS
        recall_nmhg.append(recall_at_k("gold", nmhg_ids, Config.TOP_K))
        recall_dense.append(recall_at_k("gold", dense_ids, Config.TOP_K))

        mrr_nmhg.append(mrr_at_k("gold", nmhg_ids, Config.TOP_K))
        mrr_dense.append(mrr_at_k("gold", dense_ids, Config.TOP_K))

        prec_nmhg.append(precision_at_k("gold", nmhg_ids, Config.TOP_K))
        prec_dense.append(precision_at_k("gold", dense_ids, Config.TOP_K))

        ndcg_nmhg.append(ndcg("gold", nmhg_ids, Config.TOP_K))
        ndcg_dense.append(ndcg("gold", dense_ids, Config.TOP_K))

    # ===========================
    # FINAL REPORT
    # ===========================
    print(f"{Fore.GREEN}\nFINAL PERFORMANCE METRICS")
    print("=======================================")

    print(f"Recall@{Config.TOP_K}: NMHG={np.mean(recall_nmhg):.4f}, Dense={np.mean(recall_dense):.4f}")
    print(f"MRR@{Config.TOP_K}:    NMHG={np.mean(mrr_nmhg):.4f}, Dense={np.mean(mrr_dense):.4f}")
    print(f"Precision@{Config.TOP_K}: NMHG={np.mean(prec_nmhg):.4f}, Dense={np.mean(prec_dense):.4f}")
    print(f"NDCG@{Config.TOP_K}:   NMHG={np.mean(ndcg_nmhg):.4f}, Dense={np.mean(ndcg_dense):.4f}")

    print("\nFILTERING / EFFICIENCY")
    print("---------------------------------------")
    print(f"Avg Pruning Ratio:      {np.mean(prune_ratios):.3f}")
    print(f"Avg False Negative Rate: {np.mean(fn_rates):.3f}")

    print("\nLATENCY")
    print("---------------------------------------")
    print(f"NMHG Latency (ms):       {np.mean(nmhg_latencies)*1000:.2f}")
    print(f"Dense Latency (ms):      {np.mean(dense_latencies)*1000:.2f}")
    print(f"Speedup Factor:          {np.mean(dense_latencies)/np.mean(nmhg_latencies):.2f}x")

    print(f"{Fore.GREEN}\nEvaluation complete.")


if __name__ == "__main__":
    run_benchmark()