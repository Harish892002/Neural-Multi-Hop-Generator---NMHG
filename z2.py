import time
import numpy as np
import sys
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from colorama import Fore, Style, init

# Initialize colors for the UI
init(autoreset=True)

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # [IMPORTANT] Your API Key
    PINECONE_API_KEY = "pcsk_6xEYAy_R7Bo6KKEYQGeSLtxLGnhcHEWgEM2mrK2rYUDxT8QaxTUmNAyozKukpQu8SiDhXp"
    
    # Choose Dataset: "hotpot_qa" OR "musique"
    TARGET_DATASET = "musique"
    
    # Index Map (Matches Part 1 Ingestion)
    INDEX_MAP = {
        "hotpot_qa": "nmhg-hotpotqa",
        "musique": "nmhg-musique"
    }
    
    MODEL_NAME = 'intfloat/multilingual-e5-large'
    SIMHASH_BITS = 64
    NUM_BLOCKS = 4

# ==========================================
# 2. CLIENT-SIDE LOGIC
# ==========================================
def get_query_blocks(vector, hyperplanes):
    """
    Computes the SimHash blocks for the query vector.
    This acts as the 'Keys' we use to unlock only specific parts of the database.
    """
    # Project & Binarize
    dot_prod = np.dot(vector, hyperplanes)
    full_hash = "".join(['1' if x > 0 else '0' for x in dot_prod])
    
    # Split into 4 blocks
    block_size = Config.SIMHASH_BITS // Config.NUM_BLOCKS
    blocks = []
    for i in range(Config.NUM_BLOCKS):
        start = i * block_size
        segment = full_hash[start : start + block_size]
        blocks.append(f"{i}:{segment}")
    return blocks

# ==========================================
# 3. LIVE RACE ENGINE
# ==========================================
def start_live_race():
    # --- CONNECT ---
    target_index = Config.INDEX_MAP.get(Config.TARGET_DATASET)
    print(f"{Fore.CYAN}Connecting to Index: {target_index}...")
    
    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    
    # Validation
    if target_index not in [i.name for i in pc.list_indexes()]:
        print(f"{Fore.RED}Error: Index '{target_index}' not found. Please run Part 1 (Ingest) first.")
        return

    index = pc.Index(target_index)
    
    print(f"{Fore.CYAN}Loading AI Model ({Config.MODEL_NAME})...")
    model = SentenceTransformer(Config.MODEL_NAME)
    
    # Restore Hyperplanes (Seed 42 is critical to match Database)
    np.random.seed(42) 
    hyperplanes = np.random.randn(1024, Config.SIMHASH_BITS)

    # --- UI HEADER ---
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f" NMHG LIVE BATTLE ARENA")
    print(f" Dataset: {Config.TARGET_DATASET.upper()}")
    print(f"{'='*60}")
    print(f"{Fore.WHITE}Type 'exit' to quit.\n")

    # --- MAIN LOOP ---
    while True:
        user_query = input(f"{Fore.YELLOW}Enter Question: {Fore.WHITE}")
        
        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
        
        if not user_query.strip(): continue

        print(f"{Fore.CYAN}Encoding query and running race...")
        
        # 1. Encode Query
        # E5 requires 'query: ' prefix
        q_vec = model.encode(f"query: {user_query}", normalize_embeddings=True).tolist()

        # ---------------------------------------------------------
        # COMPETITOR A: STANDARD SEARCH (BASELINE)
        # ---------------------------------------------------------
        # We query Pinecone WITHOUT filters. It scans everything.
        start = time.time()
        res_base = index.query(
            vector=q_vec, 
            top_k=5, 
            include_metadata=True
        )
        t_base = time.time() - start

        # ---------------------------------------------------------
        # COMPETITOR B: NMHG SEARCH (OURS)
        # ---------------------------------------------------------
        # We query Pinecone WITH SimHash filters. It scans only matching blocks.
        start = time.time()
        
        # Calculate blocks (<1ms local computation)
        q_blocks = get_query_blocks(q_vec, hyperplanes)
        
        res_nmhg = index.query(
            vector=q_vec, 
            top_k=5, 
            include_metadata=True,
            filter={"simhash_blocks": {"$in": q_blocks}}
        )
        t_nmhg = time.time() - start

        # ---------------------------------------------------------
        # RESULTS DISPLAY
        # ---------------------------------------------------------
        speedup = t_base / t_nmhg if t_nmhg > 0 else 1.0
        
        print(f"\n{Fore.WHITE}{'-'*60}")
        print(f"{Fore.WHITE}PERFORMANCE METRICS")
        print(f"{Fore.WHITE}{'-'*60}")
        print(f"Standard Latency:  {t_base:.4f}s")
        print(f"NMHG Latency:      {t_nmhg:.4f}s")
        print(f"SPEEDUP FACTOR:    {Fore.GREEN}{speedup:.1f}x FASTER")
        print(f"{Fore.WHITE}{'-'*60}")

        print(f"\n{Fore.CYAN}RETRIEVED DOCUMENTS (Top 5):")
        if not res_nmhg['matches']:
            print(f"{Fore.RED}No documents found matching the filter.")
        
        for i, match in enumerate(res_nmhg['matches'][:5]):
            score = match['score']
            text = match['metadata'].get('text', 'No Text')
            # Truncate text for cleaner display
            clean_text = (text[:120] + '...') if len(text) > 120 else text
            print(f"{Fore.YELLOW}[{i+1}] {Fore.BLUE}(Sim: {score:.3f}) {Fore.WHITE}{clean_text}")
        
        print("\n")

if __name__ == "__main__":
    start_live_race()