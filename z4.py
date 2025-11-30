# z4_graph_and_retriever.py
#
# NMHG: Graph construction + Synthetic Multimodal Dataset + Hop Planner + Evidence Verifier
#
# This file is self-contained and does NOT depend on Pinecone.
# It builds an in-memory retrieval graph (SPRG) as described in the mid-report:
#   - Nodes: text chunks + synthetic "image" nodes with captions
#   - Embeddings: multilingual-e5-large (text) + CLIP ViT-B/32 (image captions)
#   - Edges: cosine similarity + topic-based caption<->image links
#   - SimHash signatures for hash-guided pruning
#   - Hop Planner: multi-hop beam search with SimHash pruning
#   - Evidence Verifier: heuristic chain scoring

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import open_clip


# =========================
# 1. CONFIG
# =========================

@dataclass
class EmbeddingConfig:
    text_model_name: str = "intfloat/multilingual-e5-large"
    image_model_name: str = "ViT-B-32"
    image_pretrained: str = "openai"
    device: str = "mps"  # "cuda", "mps", or "cpu"


@dataclass
class GraphConfig:
    simhash_bits: int = 64
    hamming_threshold: int = 16
    max_neighbors: int = 150
    similarity_edge_threshold: float = 0.15  # cosine
    max_nodes_debug: Optional[int] = None  # optional cap


@dataclass
class RetrievalConfig:
    max_hops: int = 3
    beam_width: int = 20
    top_k_start_nodes: int = 30
    top_k_per_hop: int = 10


@dataclass
class NMHGConfig:
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


# =========================
# 2. SYNTHETIC MULTIMODAL DATASET
# =========================

@dataclass
class Node:
    node_id: int
    modality: str  # "text" | "image"
    text: Optional[str]
    image_id: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class QAItem:
    question: str
    answer: str
    supporting_node_ids: List[int]


class SyntheticMultimodalMiniGraph:
    """
    Synthetic dataset:
      - ~1500 text chunks
      - ~200 image nodes with captions
      - topic tags + synthetic hop labels
      - QA pairs with 2–3 supporting nodes
    """

    def __init__(
        self,
        num_text_chunks: int = 1500,
        num_images: int = 200,
        seed: int = 42,
    ) -> None:
        self.rng = random.Random(seed)
        self.num_text_chunks = num_text_chunks
        self.num_images = num_images
        self.nodes: List[Node] = []
        self.qa_items: List[QAItem] = []

        self._build_nodes()
        self._build_qa_pairs()

    def _build_nodes(self) -> None:
        topics = ["physics", "history", "biology", "cs", "art", "law"]

        # text nodes
        for i in range(self.num_text_chunks):
            topic = self.rng.choice(topics)
            hop = i % 3
            text = (
                f"Text chunk {i} about {topic}. "
                f"This node approximates hop {hop} in a synthetic reasoning chain."
            )
            self.nodes.append(
                Node(
                    node_id=i,
                    modality="text",
                    text=text,
                    image_id=None,
                    metadata={"topic": topic, "hop": hop},
                )
            )

        # image nodes with captions
        base_id = self.num_text_chunks
        for j in range(self.num_images):
            node_id = base_id + j
            topic = self.rng.choice(topics)
            caption = (
                f"Image {j} illustrating {topic} in the synthetic multimodal dataset. "
                f"This serves as visual evidence for chains about {topic}."
            )
            self.nodes.append(
                Node(
                    node_id=node_id,
                    modality="image",
                    text=caption,
                    image_id=f"img_{j}",
                    metadata={
                        "topic": topic,
                        "is_image": True,
                        "caption_to_image_chain": True,
                    },
                )
            )

    def _build_qa_pairs(self, num_pairs: int = 50) -> None:
        topics = ["physics", "history", "biology", "cs", "art", "law"]
        for _ in range(num_pairs):
            topic = self.rng.choice(topics)
            candidate_nodes = [n for n in self.nodes if n.metadata.get("topic") == topic]
            self.rng.shuffle(candidate_nodes)
            # Step 1: restrict to nodes of the same topic
            topic_nodes = [n for n in self.nodes if n.metadata.get("topic") == topic]

            # Step 2: sort them by node_id (ensures sequential semantic locality)
            topic_nodes.sort(key=lambda n: n.node_id)

            # Step 3: pick a random starting point
            if len(topic_nodes) < 3:
                continue
            
            start_idx = self.rng.randint(0, len(topic_nodes) - 3)

            # Step 4: pick 2 or 3 consecutive nodes
            chain_len = self.rng.randint(2, 4)
            chain = topic_nodes[start_idx:start_idx + chain_len]
            if len(chain) < 2:
                continue
            q = (
                f"In our synthetic world, explain the concept of {topic} using "
                f"relevant evidence."
            )
            a = (
                f"The answer discusses {topic} by combining {len(chain)} evidence nodes "
                f"from the synthetic multimodal graph."
            )
            self.qa_items.append(
                QAItem(
                    question=q,
                    answer=a,
                    supporting_node_ids=[n.node_id for n in chain],
                )
            )

    def get_nodes(self) -> List[Node]:
        return self.nodes

    def get_qa_items(self) -> List[QAItem]:
        return self.qa_items


# =========================
# 3. EMBEDDING BACKEND
# =========================

class EmbeddingBackend:
    """
    E5 for text, CLIP for images (using captions for synthetic data).
    """

    def __init__(self, cfg: NMHGConfig) -> None:
        self.cfg = cfg
        self.device = cfg.embeddings.device

        # text encoder
        self.text_model = SentenceTransformer(
            cfg.embeddings.text_model_name, device=self.device
        )

        # CLIP encoder
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            cfg.embeddings.image_model_name,
            pretrained=cfg.embeddings.image_pretrained,
            device=self.device,
        )
        self.clip_tokenizer = open_clip.get_tokenizer(cfg.embeddings.image_model_name)

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        emb = self.text_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype("float32")

    @torch.no_grad()
    def embed_captions_as_images(self, captions: List[str]) -> np.ndarray:
        tokens = self.clip_tokenizer(captions).to(self.device)
        feats = self.clip_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")


# =========================
# 4. SIMHASH UTILITIES
# =========================

def simhash_vector(vec: np.ndarray, n_bits: int = 64, rng_seed: int = 42) -> int:
    rng = np.random.default_rng(rng_seed)
    dim = vec.shape[0]
    hyperplanes = rng.standard_normal(size=(n_bits, dim))
    projections = hyperplanes @ vec
    bits = (projections > 0).astype(np.uint64)
    value = np.uint64(0)
    for i, b in enumerate(bits):
        if b:
            value |= (np.uint64(1) << np.uint64(i))
    return int(value)


def simhash_batch(mat: np.ndarray, n_bits: int = 64, base_seed: int = 42) -> List[int]:
    hashes: List[int] = []
    for i, v in enumerate(mat):
        h = simhash_vector(v, n_bits=n_bits, rng_seed=base_seed + i)
        hashes.append(h)
    return hashes


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# =========================
# 5. GRAPH STRUCTURE
# =========================

@dataclass
class GraphNode:
    node_id: int
    modality: str
    embedding: np.ndarray
    simhash: int
    metadata: Dict[str, Any]


@dataclass
class GraphEdge:
    src: int
    dst: int
    weight: float
    relation: str


@dataclass
class RetrievalGraph:
    nodes: Dict[int, GraphNode] = field(default_factory=dict)
    adj: Dict[int, List[GraphEdge]] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self.adj:
            self.adj[node.node_id] = []

    def add_edge(self, src: int, dst: int, weight: float, relation: str) -> None:
        self.adj.setdefault(src, []).append(
            GraphEdge(src=src, dst=dst, weight=weight, relation=relation)
        )

    def neighbors(self, node_id: int) -> List[GraphEdge]:
        return self.adj.get(node_id, [])


class GraphBuilder:
    """
    Build the SimHash-Guided Probabilistic Retrieval Graph (SPRG).
    """

    def __init__(self, cfg: NMHGConfig, embed_backend: EmbeddingBackend) -> None:
        self.cfg = cfg
        self.embed_backend = embed_backend

    def build_from_synthetic(self, dataset: SyntheticMultimodalMiniGraph) -> RetrievalGraph:
        cfg = self.cfg
        graph = RetrievalGraph()
        nodes = dataset.get_nodes()

        text_nodes = [n for n in nodes if n.modality == "text"]
        image_nodes = [n for n in nodes if n.modality == "image"]

        # embeddings
        text_embs = self.embed_backend.embed_texts([n.text or "" for n in text_nodes])
        img_embs = self.embed_backend.embed_captions_as_images(
            [n.text or "" for n in image_nodes]
        )

        text_dim = text_embs.shape[1]
        img_dim = img_embs.shape[1]

        if img_dim != text_dim:
            W = np.random.normal(0, 0.05, size=(img_dim, text_dim)).astype("float32")
            img_embs = img_embs @ W     # shape: (num_images, 1024)
            # Normalize after projection
            img_embs = img_embs / (np.linalg.norm(img_embs, axis=1, keepdims=True) + 1e-9)

        # simhash
        text_hashes = simhash_batch(text_embs, n_bits=cfg.graph.simhash_bits, base_seed=42)
        img_hashes = simhash_batch(img_embs, n_bits=cfg.graph.simhash_bits, base_seed=1337)

        # add nodes (also store raw text in metadata for generator later)
        for n, e, h in zip(text_nodes, text_embs, text_hashes):
            m = dict(n.metadata)
            m["raw_text"] = n.text or ""
            graph.add_node(
                GraphNode(
                    node_id=n.node_id,
                    modality=n.modality,
                    embedding=e,
                    simhash=h,
                    metadata=m,
                )
            )

        for n, e, h in zip(image_nodes, img_embs, img_hashes):
            m = dict(n.metadata)
            m["raw_text"] = n.text or ""
            graph.add_node(
                GraphNode(
                    node_id=n.node_id,
                    modality=n.modality,
                    embedding=e,
                    simhash=h,
                    metadata=m,
                )
            )

        # similarity edges
        all_ids = list(graph.nodes.keys())
        embs = np.stack([graph.nodes[i].embedding for i in all_ids], axis=0)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        sim_matrix = embs @ embs.T
        n_nodes = len(all_ids)
        max_neighbors = cfg.graph.max_neighbors
        thr = cfg.graph.similarity_edge_threshold

        for i in range(n_nodes):
            src_id = all_ids[i]
            sims = sim_matrix[i]
            neighbor_idx = np.argsort(-sims)[: max_neighbors + 1]
            for j in neighbor_idx:
                if i == j:
                    continue
                dst_id = all_ids[j]
                sim = float(sims[j])
                if sim < thr:
                    continue
                graph.add_edge(src_id, dst_id, sim, "cosine_similarity")

        # caption-image alignment by topic
        topic_to_imgs: Dict[str, List[int]] = {}
        for n in image_nodes:
            topic_to_imgs.setdefault(n.metadata.get("topic"), []).append(n.node_id)

        for n in text_nodes:
            topic = n.metadata.get("topic")
            img_ids = topic_to_imgs.get(topic, [])
            for img_id in img_ids:
                graph.add_edge(n.node_id, img_id, 1.0, "caption_image")
                graph.add_edge(img_id, n.node_id, 1.0, "image_caption")

        return graph


# =========================
# 6. HOP PLANNER
# =========================

@dataclass
class HopState:
    node_id: int
    score: float
    path: List[int]


class HopPlanner:
    """
    Multi-hop retrieval over the SPRG using:
      - beam search
      - SimHash-based pruning
      - cosine similarity to the question embedding
    """

    def __init__(self, cfg: NMHGConfig, graph: RetrievalGraph) -> None:
        self.cfg = cfg
        self.graph = graph

        self.node_ids: List[int] = list(graph.nodes.keys())
        self.node_hashes: List[int] = [graph.nodes[i].simhash for i in self.node_ids]
        embs = np.stack(
            [graph.nodes[i].embedding for i in self.node_ids], axis=0
        ).astype("float32")
        self.emb_matrix = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    def _dense_similarity(self, q_emb: np.ndarray) -> np.ndarray:
        q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        return self.emb_matrix @ q

    def _start_states(self, q_emb: np.ndarray) -> List[HopState]:
        sims = self._dense_similarity(q_emb)
        k = self.cfg.retrieval.top_k_start_nodes
        top_idx = np.argsort(-sims)[:k]
        states: List[HopState] = []
        for idx in top_idx:
            nid = self.node_ids[idx]
            score = float(sims[idx])
            states.append(HopState(node_id=nid, score=score, path=[nid]))
        return states

    def _prune_by_simhash(self, q_hash: int, candidate_ids: List[int]) -> List[int]:
        pruned: List[int] = []
        for cid in candidate_ids:
            idx = self.node_ids.index(cid)
            h = self.node_hashes[idx]
            if hamming_distance(q_hash, h) <= self.cfg.graph.hamming_threshold:
                pruned.append(cid)
        return pruned

    def plan_hops(self, q_emb: np.ndarray, q_hash: int) -> List[HopState]:
        cfg = self.cfg
        beam = self._start_states(q_emb)

        for _ in range(cfg.retrieval.max_hops - 1):
            new_beam: List[HopState] = []
            for state in beam:
                neigh_edges = self.graph.neighbors(state.node_id)
                cand_ids = [e.dst for e in neigh_edges]
                if not cand_ids:
                    continue
                pruned_ids = self._prune_by_simhash(q_hash, cand_ids)
                if not pruned_ids:
                    continue

                q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
                sims = []
                for cid in pruned_ids:
                    emb = self.graph.nodes[cid].embedding
                    emb = emb / (np.linalg.norm(emb) + 1e-9)
                    sims.append(float(emb @ q))

                sims_arr = np.array(sims, dtype="float32")
                idx_sorted = np.argsort(-sims_arr)[: cfg.retrieval.top_k_per_hop]

                for i in idx_sorted:
                    cid = pruned_ids[i]
                    score = state.score + float(sims_arr[i])
                    new_state = HopState(
                        node_id=cid,
                        score=score,
                        path=state.path + [cid],
                    )
                    new_beam.append(new_state)

            if not new_beam:
                break
            new_beam = sorted(new_beam, key=lambda s: s.score, reverse=True)[
                : cfg.retrieval.beam_width
            ]
            beam = new_beam

        return beam


# =========================
# 7. EVIDENCE VERIFIER
# =========================

class EvidenceVerifier:
    """
    Heuristic verifier:
      - mean similarity of nodes in chain to question
      - smoothness of node-to-node transitions
    """

    def __init__(self, graph: RetrievalGraph) -> None:
        self.graph = graph

    def verify_chain(self, q_emb: np.ndarray, node_ids: List[int]) -> float:
        if not node_ids:
            return 0.0
        q = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        embs = []
        for nid in node_ids:
            e = self.graph.nodes[nid].embedding
            e = e / (np.linalg.norm(e) + 1e-9)
            embs.append(e)
        embs = np.stack(embs, axis=0)

        q_sims = embs @ q
        mean_q_sim = float(q_sims.mean())

        if len(node_ids) > 1:
            pair_sims = []
            for i in range(len(node_ids) - 1):
                pair_sims.append(float(embs[i] @ embs[i + 1]))
            smoothness = float(np.mean(pair_sims))
        else:
            smoothness = 0.0

        score = 0.7 * mean_q_sim + 0.3 * smoothness
        return score


# =========================
# 8. SYSTEM CONSTRUCTOR
# =========================

def build_nmhg_system() -> Tuple[
    NMHGConfig,
    EmbeddingBackend,
    SyntheticMultimodalMiniGraph,
    RetrievalGraph,
    HopPlanner,
    EvidenceVerifier,
]:
    """
    Convenience function used by z5_generator_and_eval.py.
    """
    cfg = NMHGConfig()
    embed_backend = EmbeddingBackend(cfg)
    dataset = SyntheticMultimodalMiniGraph()
    graph_builder = GraphBuilder(cfg, embed_backend)
    graph = graph_builder.build_from_synthetic(dataset)
    hop_planner = HopPlanner(cfg, graph)
    verifier = EvidenceVerifier(graph)
    return cfg, embed_backend, dataset, graph, hop_planner, verifier


if __name__ == "__main__":
    # Minimal smoke test
    cfg, backend, dataset, graph, hp, ver = build_nmhg_system()
    print("NMHG system constructed.")
    print(f"Nodes in graph: {len(graph.nodes)}")
    print(f"QA items: {len(dataset.get_qa_items())}")