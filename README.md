# NMHG: Neural Multi-Hop Generator  
### *A Scalable Architecture for Multi-Hop Retrieval, Evidence Verification, and Generative Reasoning*

This repository implements **NMHG**, a full-stack multi-hop retrieval & reasoning system composed of:

1. **A Pinecone-backed scalable retrieval engine** using  
   - multilingual-e5-large embeddings  
   - SimHash-guided server-side filtering  
   - multimodal (text + synthetic images) ingestion  
   - real-dataset evaluation (HotpotQA, MuSiQue, M-BEIR)

2. **A local architectural NMHG engine** demonstrating  
   - synthetic multimodal graph  
   - hop planner (beam search)  
   - SimHash-guided pruning  
   - evidence verifier  
   - evidence-constrained generator  
   - multi-hop QA chain evaluation  

Together, these modules deliver a complete end-to-end system aligned with all mid-report and final project requirements.


# Repository Structure
```
.
├── z1.py   # Pinecone ingestion: HotpotQA + MuSiQue + synthetic multimodal nodes
├── z2.py   # Live retrieval UI (baseline vs NMHG filter) using Pinecone indexes
├── z3.py   # M-BEIR benchmark: Recall, MRR, NDCG, pruning rate, latency, speedup
├── z4.py   # Local NMHG graph: embeddings, SimHash, edges, hop planner, verifier
├── z5.py   # End-to-end local NMHG multi-hop reasoning + generator + baselines
└── README.md   # (this file)
```

# System Overview

NMHG is organized into **two pillars**, each serving a distinct purpose:


## Pillar A — Real-World Scalable Retrieval (z1–z2–z3)

These components demonstrate NMHG’s scalability on real datasets using Pinecone.

### Datasets:
- **HotpotQA (validation split)**
- **MuSiQue (validation split)**
- **M-BEIR (query split)**

### Features:
- Vector ingestion into Pinecone (text + synthetic images)
- SimHash block-based metadata for filtering
- Real retrieval comparisons:
  - baseline dense retrieval  
  - NMHG filtered retrieval  
- Speedup measurement
- Full evaluation metrics on M-BEIR:
  - Recall@10  
  - MRR@10  
  - Precision@10  
  - NDCG@10  
  - Latency  
  - Pruning ratio  
  - False-negative rate  
  - Speedup factor  

These scripts provide the **quantitative** and **scalability** results for the project.


## Pillar B — NMHG Reasoning Engine (z4–z5)

These files demonstrate the **architectural behavior** of NMHG using a synthetic multimodal world.

### Features:
- Synthetic graph: ~1500 text nodes + ~200 projected image nodes
- E5 text embeddings + CLIP image embeddings (projected to 1024-dim)
- Cosine similarity edges + caption-image edges
- SimHash-based pruning structure
- Hop planner using beam search
- Evidence verifier scoring chains
- Synthetic QA dataset with gold chains
- Evidence-only generator
- Baselines:
  - Dense kNN retrieval (NumPy)
  - Query reformulation baseline

Provides qualitative examples and demonstrates **multi-hop chain reasoning**.


# Detailed File Descriptions

## z1.py — Pinecone Ingestion Pipeline

This script handles:

- Loading HotpotQA + MuSiQue
- Constructing text passages
- Generating 5000 synthetic “image nodes”
- Embedding text using **multilingual-e5-large**
- Embedding images using **CLIP ViT-B/32** then projecting to 1024-dim
- Computing 64-bit SimHash blocks for each vector
- Upserting all data into Pinecone indexes:
  - `nmhg-hotpotqa`
  - `nmhg-musique`

### Run:
```bash
python z1.py
```

## z2.py — Live Pinecone Retrieval UI

A terminal UI for interactively comparing:
- baseline vector search (Pinecone dense)
- NMHG SimHash-filtered search

For each query, it prints:
- baseline latency
- NMHG latency
- speedup factor
- retrieved documents

### Run:
```bash
python z2.py
```

## z3.py — M-BEIR Retrieval Benchmark (Local)

Provides all retrieval metrics needed for the final report.

Computes:

### Retrieval Quality
- Recall@10
- MRR@10
- Precision@10
- NDCG@10

### Filtering Performance
- Pruning ratio
- False-negative rate

### Latency
- NMHG retrieval latency
- Dense retrieval latency
- Speedup factor

### Dataset:
- M-BEIR (query split), streaming evaluation

Run:
```bash
python z3.py
```

## z4.py — NMHG Graph Constructor

Builds the entire local NMHG architecture:
- Random multimodal graph
- Topic-aligned text nodes
- Projected CLIP image nodes
- Cosine similarity edges
- Caption-image edges
- 64-bit SimHash hashing
- Hop Planner (beam search)
- Evidence Verifier
- Synthetic QA dataset creation

This is the core architectural implementation.

## z5.py — Local NMHG Multi-Hop Reasoning Engine

Runs the full NMHG pipeline:

### ✔ For each synthetic QA pair:
- Encodes question using E5
- Computes SimHash for pruning
- Executes hop planner
- Retrieves multi-hop chain
- Verifies chain quality
- Generates answer using evidence only
- Compares against baselines
- Prints example chains and generated answers

### Computes:
- Recall@K
- Chain coverage
- Intermediate precision
- Avg nodes explored per hop

Run:
```bash
python z5.py
```
Evaluation Summary

## Local Evaluation (M-BEIR, z3)

This gives the intrinsic retrieval performance of NMHG:
- Recall@10
- Precision@10
- MRR@10
- NDCG@10
- Latency (dense vs NMHG)
- Speedup factor
- Pruning ratio
- False-negative rate

These metrics appear in the “Results: Retrieval Quality” section of the final report.


## Pinecone Evaluation (z2)

This gives the real-world scalability performance:
- Latency of baseline search
- Latency of NMHG filtered search
- Speedup factor
- Real examples from HotpotQA / MuSiQue

These results belong in the “Results: Efficiency & Scalability” section.


## Architectural Demonstration (z4–z5)

This gives the qualitative & architectural behavior:
- Synthetic QA chains
- Retrieved multi-hop paths
- Evidence-verifier scoring
- Evidence-constrained answer generation
- Baseline comparisons
- Efficiency (nodes explored per hop)

These belong in the “Methodology & Architecture” section.


# Installation & Dependencies

Install Python packages:
```bash
pip install sentence-transformers
pip install open_clip_torch
pip install numpy
pip install pinecone-client
pip install datasets
pip install colorama
pip install tqdm
```

# Citing Datasets & Models

This project uses:
- HotpotQA
- MuSiQue
- M-BEIR
- E5-Large multilingual
- CLIP ViT-B/32


# Conclusion

This repository delivers a complete end-to-end NMHG system including:
- Scalable vector retrieval
- SimHash-based filtering
- Multi-hop reasoning
- Evidence verification
- Synthetic knowledge graph
- Local + cloud evaluation
- Baselines & qualitative samples
- Full retrieval metrics
