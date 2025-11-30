import os
import time
import numpy as np
import random
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
class Config:
    # --- API ---
    PINECONE_API_KEY = "pcsk_6xEYAy_R7Bo6KKEYQGeSLtxLGnhcHEWgEM2mrK2rYUDxT8QaxTUmNAyozKukpQu8SiDhXp"
    
    # --- INDICES ---
    # We will create TWO indices as requested
    IDX_HOTPOT = "nmhg-hotpotqa"
    IDX_MUSIQUE = "nmhg-musique"
    
    # --- MODEL ---
    # Using the heavy hitter: E5-Large (1024 dim)
    MODEL_NAME = 'intfloat/multilingual-e5-large'
    EMBED_DIM = 1024
    
    # --- SIMHASH (Server-Side Filtering) ---
    SIMHASH_BITS = 64
    NUM_BLOCKS = 4  # Splits 64 bits into 4x16-bit searchable blocks
    
    # --- DATA SCALE ---
    # Set to None for "No Limits" (Loads full Validation sets)
    # WARNING: HotpotQA Val ~74k docs. MuSiQue Val ~50k docs.
    # Total ~120k docs. (Check your Pinecone Free Tier limit of 100k!)
    MAX_DOCS_PER_SET = None 
    
    # --- MULTIMODAL SCALE ---
    # How many synthetic 'Image' nodes to inject per index
    NUM_IMAGES = 5000 

# ==========================================
# 2. SIMHASH ENGINE
# ==========================================
def get_simhash_metadata(vector, hyperplanes):
    """
    Computes SimHash and splits it into blocks for Pinecone Metadata Filtering.
    """
    # 1. Project & Binarize
    dot_products = np.dot(vector, hyperplanes)
    full_hash = "".join(['1' if x > 0 else '0' for x in dot_products])
    
    # 2. Block Split (Pigeonhole Principle)
    # Allows querying: "WHERE simhash_blocks HAS '0:1011...'"
    block_size = Config.SIMHASH_BITS // Config.NUM_BLOCKS
    blocks = []
    for i in range(Config.NUM_BLOCKS):
        start = i * block_size
        end = start + block_size
        segment = full_hash[start:end]
        blocks.append(f"{i}:{segment}")
        
    return blocks

# ==========================================
# 3. DATA LOADERS (NO LIMITS)
# ==========================================
def load_hotpot_data():
    print(f"{Fore.CYAN}Fetching HotpotQA (Validation Split)...")
    # [cite: 45] HotpotQA requires reasoning over multiple paragraphs
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    
    passages = []
    # Iterate through ALL questions
    for row in tqdm(dataset, desc="Parsing HotpotQA"):
        # Hotpot structure: List of Titles, List of Sentence-Lists
        for title, sentences in zip(row['context']['title'], row['context']['sentences']):
            # We join the first few sentences to make a solid paragraph
            if sentences:
                text = f"{title}: {' '.join(sentences[:3])}"
                passages.append(text)
                
    if Config.MAX_DOCS_PER_SET:
        passages = passages[:Config.MAX_DOCS_PER_SET]
        
    print(f"{Fore.GREEN}Loaded {len(passages)} HotpotQA documents.")
    return passages

def load_musique_data():
    print(f"{Fore.CYAN}Fetching MuSiQue (Answerable Split)...")
    # [cite: 46] MuSiQue is designed for multi-hop decomposition
    dataset = load_dataset("dgslibisey/MuSiQue", split="validation", trust_remote_code=True)
    
    passages = []
    for row in tqdm(dataset, desc="Parsing MuSiQue"):
        for para in row['paragraphs']:
            text = f"{para['title']}: {para['paragraph_text']}"
            passages.append(text)
            
    if Config.MAX_DOCS_PER_SET:
        passages = passages[:Config.MAX_DOCS_PER_SET]
        
    print(f"{Fore.GREEN}Loaded {len(passages)} MuSiQue documents.")
    return passages

def generate_multimodal_nodes(count):
    """
    Generates synthetic image descriptions to scale the 'Multimodal' graph.
    [cite: 47, 51] Matches the 'Synthetic Multimodal Mini-Graph' description.
    """
    print(f"{Fore.CYAN}Generating {count} Synthetic Image Nodes...")
    subjects = ["Eiffel Tower", "Python Code", "Neural Network", "Ancient Rome", "DNA Helix", "Mars Rover"]
    types = ["Diagram", "Photograph", "Sketch", "Heatmap", "Screenshot"]
    
    nodes = []
    for _ in range(count):
        subj = random.choice(subjects)
        typ = random.choice(types)
        # E.g., "[IMAGE] Photograph of Ancient Rome ruins"
        text = f"[IMAGE] {typ} of {subj} context ID {random.randint(1000,9999)}"
        nodes.append(text)
    return nodes

# ==========================================
# 4. UNIVERSAL UPSERTER
# ==========================================
def upsert_to_pinecone(index_name, text_list, model, hyperplanes):
    # 1. Connect
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    # 2. Check/Create Index
    if index_name not in pc.list_indexes().names():
        print(f"{Fore.YELLOW}Creating Index: {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=Config.EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5) # Wait for init
    
    index = pc.Index(index_name)
    
    # 3. Batch Process
    batch_size = 50
    print(f"{Fore.WHITE}Upserting {len(text_list)} vectors to {index_name}...")
    
    for i in tqdm(range(0, len(text_list), batch_size), desc=f"Pushing to {index_name}"):
        batch_text = text_list[i : i + batch_size]
        batch_ids = [f"doc_{time.time()}_{k}" for k in range(len(batch_text))] # Unique IDs
        
        # Encode (E5 requires 'passage:' prefix)
        e5_inputs = [f"passage: {t}" for t in batch_text]
        embeddings = model.encode(e5_inputs, normalize_embeddings=True)
        
        to_upsert = []
        for j, vec in enumerate(embeddings):
            # Generate SimHash Blocks
            blocks = get_simhash_metadata(vec, hyperplanes)
            
            # Determine Node Type
            node_type = "image" if "[IMAGE]" in batch_text[j] else "text"
            
            to_upsert.append({
                "id": batch_ids[j],
                "values": vec.tolist(),
                "metadata": {
                    "text": batch_text[j],
                    "type": node_type,
                    "simhash_blocks": blocks # Crucial for NMHG retrieval!
                }
            })
            
        index.upsert(vectors=to_upsert)
        
    print(f"{Fore.GREEN}Finished ingestion for {index_name}.")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Setup Model & Hyperplanes
    print(f"{Fore.CYAN}Initializing Models...")
    model = SentenceTransformer(Config.MODEL_NAME)
    
    # FIXED SEED is critical. 
    # If you change this seed, the SimHashes will change and retrieval will fail!
    np.random.seed(42) 
    hyperplanes = np.random.randn(Config.EMBED_DIM, Config.SIMHASH_BITS)
    
    # 2. Process HotpotQA
    print(f"\n{Fore.YELLOW}{'='*40}")
    print(f" PHASE 1: HOTPOTQA INGESTION")
    print(f"{'='*40}")
    hotpot_docs = load_hotpot_data()
    # Add synthetic images to Hotpot Index
    hotpot_imgs = generate_multimodal_nodes(Config.NUM_IMAGES)
    
    upsert_to_pinecone(Config.IDX_HOTPOT, hotpot_docs + hotpot_imgs, model, hyperplanes)
    
    # 3. Process MuSiQue
    print(f"\n{Fore.YELLOW}{'='*40}")
    print(f" PHASE 2: MUSIQUE INGESTION")
    print(f"{'='*40}")
    musique_docs = load_musique_data()
    # Add synthetic images to MuSiQue Index
    musique_imgs = generate_multimodal_nodes(Config.NUM_IMAGES)
    
    upsert_to_pinecone(Config.IDX_MUSIQUE, musique_docs + musique_imgs, model, hyperplanes)

    print(f"\n{Fore.GREEN}{'='*50}")
    print(f" ALL DATA INGESTED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"Indices Created:")
    print(f"  1. {Config.IDX_HOTPOT} (Text + Multimodal)")
    print(f"  2. {Config.IDX_MUSIQUE} (Text + Multimodal)")
    print(f"SimHash Configuration:")
    print(f"  - Bits: {Config.SIMHASH_BITS}")
    print(f"  - Blocks: {Config.NUM_BLOCKS}")