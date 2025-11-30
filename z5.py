# z5_generator_and_eval.py
#
# NMHG: Generator with Evidence Hooks + Baselines + Retrieval Metrics + End-to-End Experiment
#
# This file depends on z4_graph_and_retriever.py and ties everything together:
#   - evidence-constrained generator
#   - dense kNN baseline (FAISS)
#   - query-reformulation baseline
#   - retrieval metrics (Recall@k, chain coverage, intermediate precision)
#   - efficiency metric (nodes explored per hop)
#   - qualitative examples
#
# Run:
#   python z5_generator_and_eval.py
#
# It will:
#   - build the synthetic NMHG system
#   - run multi-hop retrieval
#   - compare against baselines
#   - print metrics and example answers

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Set

import numpy as np
# import faiss
from colorama import Fore, init

from z4 import (
    build_nmhg_system,
    QAItem,
)

init(autoreset=True)


# =========================
# 1. GENERATOR WITH EVIDENCE HOOKS
# =========================

class GeneratorWithEvidenceHooks:
    """
    Evidence-constrained answer generator:
      - Accepts question + ordered chain of node IDs
      - Assembles answer strictly from node metadata/raw_text
    """

    def __init__(self, graph) -> None:
        self.graph = graph

    def _node_to_snippet(self, node_id: int) -> str:
        node = self.graph.nodes[node_id]
        topic = node.metadata.get("topic", None)
        prefix = f"[topic: {topic}] " if topic else ""
        raw = node.metadata.get("raw_text", f"Node {node_id} (no text).")
        return prefix + raw

    def generate(self, question: str, chain_node_ids: List[int]) -> str:
        if not chain_node_ids:
            return (
                f"Question: {question}\n\n"
                f"Unable to answer reliably: no evidence chain was retrieved."
            )
        snippets = [self._node_to_snippet(nid) for nid in chain_node_ids]
        joined = " ".join(snippets)
        if len(joined) > 800:
            joined = joined[:797] + "..."
        answer = (
            f"Question: {question}\n\n"
            "Answer (constructed from retrieved evidence only):\n"
            f"{joined}"
        )
        return answer


# =========================
# 2. BASELINES
# =========================

class DenseKNNBaseline:
    """
    Dense kNN baseline using pure NumPy cosine similarity — no FAISS needed.
    """

    def __init__(self, graph):
        self.graph = graph
        self.node_ids = list(graph.nodes.keys())
        self.embs = np.stack(
            [graph.nodes[i].embedding for i in self.node_ids], axis=0
        ).astype("float32")
        # Normalize for cosine similarity
        self.embs = self.embs / (np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-9)

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> List[int]:
        q = query_emb.astype("float32")
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.embs @ q  # shape (N,)
        idxs = np.argsort(-sims)[:top_k]
        return [self.node_ids[i] for i in idxs]


class QueryReformulationBaseline:
    """
    Simple baseline:
      - reformulates the query with "Step 1/2" prefixes
      - performs independent dense searches
      - merges results
    """

    def __init__(self, dense_baseline: DenseKNNBaseline, embed_backend) -> None:
        self.dense = dense_baseline
        self.embed_backend = embed_backend

    def _reformulate(self, question: str, step: int) -> str:
        return f"Step {step}: {question}"

    def search(self, question: str, num_steps: int = 2, top_k: int = 10) -> List[int]:
        all_ids: List[int] = []
        for step in range(1, num_steps + 1):
            q_ref = self._reformulate(question, step)
            q_emb = self.embed_backend.embed_texts([q_ref])[0]
            hits = self.dense.search(q_emb, top_k)
            all_ids.extend(hits)
        seen = set()
        merged: List[int] = []
        for nid in all_ids:
            if nid not in seen:
                seen.add(nid)
                merged.append(nid)
        return merged[:top_k]


# =========================
# 3. METRICS
# =========================

@dataclass
class RetrievalEvalResult:
    recall_at_k: float
    avg_chain_coverage: float
    intermediate_precision: float


def evaluate_retrieval(
    qa_items: List[QAItem],
    retrieved_node_ids_per_query: List[List[int]],
    k: int,
) -> RetrievalEvalResult:
    assert len(qa_items) == len(retrieved_node_ids_per_query)
    total_recall = 0.0
    total_cov = 0.0
    total_iprec = 0.0
    n = len(qa_items)
    if n == 0:
        return RetrievalEvalResult(0.0, 0.0, 0.0)

    for qa, retrieved in zip(qa_items, retrieved_node_ids_per_query):
        support: Set[int] = set(qa.supporting_node_ids)
        retrieved_k: Set[int] = set(retrieved[:k])

        if not support:
            continue

        # recall@k over gold support
        inter = support.intersection(retrieved_k)
        total_recall += len(inter) / len(support)

        # chain coverage over entire retrieved list
        inter_all = support.intersection(set(retrieved))
        total_cov += len(inter_all) / len(support)

        # intermediate precision: fraction of retrieved@k that are in support
        if retrieved_k:
            total_iprec += len(inter) / len(retrieved_k)
        else:
            total_iprec += 0.0

    return RetrievalEvalResult(
        recall_at_k=total_recall / n,
        avg_chain_coverage=total_cov / n,
        intermediate_precision=total_iprec / n,
    )


@dataclass
class EfficiencyStats:
    total_hops: int
    total_nodes_explored: int
    total_queries: int

    @property
    def avg_nodes_per_hop(self) -> float:
        if self.total_hops == 0:
            return 0.0
        return self.total_nodes_explored / self.total_hops


# =========================
# 4. END-TO-END EXPERIMENT
# =========================

def run_synthetic_experiment() -> None:
    print(f"{Fore.CYAN}Building NMHG system (graph, embeddings, hop planner, verifier)...")
    cfg, embed_backend, dataset, graph, hop_planner, verifier = build_nmhg_system()

    qa_items = dataset.get_qa_items()
    print(f"{Fore.GREEN}Graph has {len(graph.nodes)} nodes; QA set has {len(qa_items)} questions.\n")

    generator = GeneratorWithEvidenceHooks(graph)
    dense_baseline = DenseKNNBaseline(graph)
    qref_baseline = QueryReformulationBaseline(dense_baseline, embed_backend)

    questions = [qa.question for qa in qa_items]
    q_embs = embed_backend.embed_texts(questions)
    # SimHash for queries uses same parameters as in z4
    from z4 import simhash_vector, NMHGConfig
    q_hashes = [simhash_vector(e, n_bits=cfg.graph.simhash_bits) for e in q_embs]

    multi_retrieved: List[List[int]] = []
    dense_retrieved: List[List[int]] = []
    qref_retrieved: List[List[int]] = []

    total_nodes_explored = 0
    total_hops = 0

    print(f"{Fore.CYAN}Running multi-hop retrieval and baselines on synthetic QA items...\n")

    for idx, qa in enumerate(qa_items):
        q_emb = q_embs[idx]
        q_hash = q_hashes[idx]

        # ---- NMHG multi-hop + verifier ----
        t0 = time.time()
        beam_states = hop_planner.plan_hops(q_emb, q_hash)
        t1 = time.time()

        if beam_states:
            rescored = [
                (state, verifier.verify_chain(q_emb, state.path))
                for state in beam_states
            ]
            rescored.sort(key=lambda t: t[1], reverse=True)
            best_state, best_verif = rescored[0]
            chain = best_state.path
        else:
            chain = []
            best_verif = 0.0

        multi_retrieved.append(chain)

        explored = set()
        for s in beam_states:
            explored.update(s.path)
        total_nodes_explored += len(explored)
        total_hops += cfg.retrieval.max_hops

        # ---- Dense baseline ----
        dense_nodes = dense_baseline.search(q_emb, top_k=cfg.retrieval.top_k_per_hop)
        dense_retrieved.append(dense_nodes)

        # ---- Query-reform baseline ----
        qref_nodes = qref_baseline.search(
            qa.question,
            num_steps=2,
            top_k=cfg.retrieval.top_k_per_hop,
        )
        qref_retrieved.append(qref_nodes)

        # ---- Example prints ----
        if idx < 3:
            print(f"{Fore.WHITE}{'='*70}")
            print(f"{Fore.YELLOW}Example {idx}")
            print(f"{Fore.CYAN}Question: {qa.question}")
            print(f"{Fore.CYAN}Gold support node IDs: {qa.supporting_node_ids}")
            print(f"{Fore.GREEN}NMHG chain: {chain}")
            print(f"{Fore.GREEN}Verifier score: {best_verif:.4f}")
            print(f"{Fore.MAGENTA}Dense baseline nodes: {dense_nodes}")
            print(f"{Fore.MAGENTA}Query-reform baseline nodes: {qref_nodes}")
            print(f"{Fore.BLUE}Multi-hop retrieval latency: {t1 - t0:.4f}s")

            answer = generator.generate(qa.question, chain)
            print(f"{Fore.WHITE}\nGenerated Answer (Evidence-Constrained):\n{answer}\n")

    # ---- Metrics ----
    k = cfg.retrieval.top_k_per_hop

    print(f"\n{Fore.CYAN}Evaluating retrieval metrics for NMHG multi-hop retriever...")
    res_multi = evaluate_retrieval(qa_items, multi_retrieved, k)
    print(
        f"{Fore.GREEN}NMHG Multi-hop -> Recall@{k}: {res_multi.recall_at_k:.4f}, "
        f"ChainCoverage: {res_multi.avg_chain_coverage:.4f}, "
        f"IntermediatePrecision: {res_multi.intermediate_precision:.4f}"
    )

    print(f"\n{Fore.CYAN}Evaluating retrieval metrics for dense kNN baseline...")
    res_dense = evaluate_retrieval(qa_items, dense_retrieved, k)
    print(
        f"{Fore.YELLOW}Dense kNN -> Recall@{k}: {res_dense.recall_at_k:.4f}, "
        f"ChainCoverage: {res_dense.avg_chain_coverage:.4f}, "
        f"IntermediatePrecision: {res_dense.intermediate_precision:.4f}"
    )

    print(f"\n{Fore.CYAN}Evaluating retrieval metrics for query reformulation baseline...")
    res_qref = evaluate_retrieval(qa_items, qref_retrieved, k)
    print(
        f"{Fore.MAGENTA}Query Reformulation -> Recall@{k}: {res_qref.recall_at_k:.4f}, "
        f"ChainCoverage: {res_qref.avg_chain_coverage:.4f}, "
        f"IntermediatePrecision: {res_qref.intermediate_precision:.4f}"
    )

    eff = EfficiencyStats(
        total_hops=total_hops,
        total_nodes_explored=total_nodes_explored,
        total_queries=len(qa_items),
    )
    print(
        f"\n{Fore.CYAN}Efficiency: "
        f"Avg nodes explored per hop = {eff.avg_nodes_per_hop:.2f}"
    )


if __name__ == "__main__":
    run_synthetic_experiment()