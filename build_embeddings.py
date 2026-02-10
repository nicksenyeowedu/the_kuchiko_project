"""
Build FAISS Embeddings Index for Hybrid RAG Chatbot
- Extracts all entities from Memgraph knowledge graph
- Generates embeddings using NVIDIA NIM
- Builds FAISS index for fast vector search
- Saves index to disk for reuse

Run this script once after building your knowledge graph with createKG.py

Install:
    pip install openai neo4j numpy faiss-cpu

Run:
    python build_embeddings.py

Output Files:
    - faiss_index.bin: FAISS vector index
    - entities.pkl: List of entity names
    - entity_metadata.pkl: Entity metadata (labels, properties)
"""

import os
import json
import logging
import pickle
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import faiss
from neo4j import GraphDatabase, basic_auth
from openai import OpenAI
from token_tracker import tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --------- Configuration ---------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    logger.error("NVIDIA_API_KEY not set. Please set it in your .env file.")
    exit(1)

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "memgraph")
MEMGRAPH_PASS = os.getenv("MEMGRAPH_PASS", "memgraph")

# Model selection
NIM_EMBEDDING_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
NIM_API_BASE = "https://integrate.api.nvidia.com/v1"

# Output directory and files
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data")
INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss_index.bin")
ENTITIES_PATH = os.path.join(OUTPUT_DIR, "entities.pkl")
METADATA_PATH = os.path.join(OUTPUT_DIR, "entity_metadata.pkl")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Batch size for embedding generation
BATCH_SIZE = 50

# Initialize OpenAI client for NVIDIA NIM
nim_client = OpenAI(
    base_url=NIM_API_BASE,
    api_key=NVIDIA_API_KEY
)

# Neo4j driver for Memgraph
driver = GraphDatabase.driver(MEMGRAPH_URI, auth=basic_auth(MEMGRAPH_USER, MEMGRAPH_PASS))


# --------- Embedding Functions ---------

def get_embedding(text: str, input_type: str = "passage") -> List[float]:
    """Get embedding vector from NVIDIA NIM.

    Args:
        text: Text to embed
        input_type: 'query' or 'passage'
    """
    try:
        t0 = time.time()
        response = nim_client.embeddings.create(
            input=[text],
            model=NIM_EMBEDDING_MODEL,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "NONE"}
        )
        elapsed_ms = (time.time() - t0) * 1000
        tracker.log_embedding([text], response=response, caller="build_embeddings.get_embedding",
                              elapsed_ms=elapsed_ms)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding for '{text[:50]}...': {e}")
        return None


def get_embeddings_batch(texts: List[str], input_type: str = "passage", max_retries: int = 3) -> List[List[float]]:
    """Get embeddings for multiple texts in a single API call with retry logic.
    
    Args:
        texts: List of texts to embed
        input_type: 'query' or 'passage'
        max_retries: Maximum number of retry attempts
    
    Returns:
        List of embedding vectors
    """
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = nim_client.embeddings.create(
                input=texts,
                model=NIM_EMBEDDING_MODEL,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "NONE"}
            )
            elapsed_ms = (time.time() - t0) * 1000
            tracker.log_embedding(texts, response=response, caller="build_embeddings.get_embeddings_batch",
                                  elapsed_ms=elapsed_ms)
            return [item.embedding for item in response.data]
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                    logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    return None
            else:
                logger.error(f"Error getting batch embeddings: {e}")
                return None
    return None


def extract_entities_from_kg() -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Extract all entities from knowledge graph (excluding chat history).
    
    Returns:
        entity_names: List of entity names
        entity_texts: List of texts to embed (name + description + type)
        entity_metadata: List of metadata dicts
    """
    logger.info("Extracting entities from Memgraph...")
    
    entity_names = []
    entity_texts = []
    entity_metadata = []
    
    try:
        with driver.session() as session:
            # Query all entities (exclude User and Message nodes for chat history)
            query = """
            MATCH (n)
            WHERE NOT n:User AND NOT n:Message
            AND (n.name IS NOT NULL OR n.id IS NOT NULL OR n.title IS NOT NULL)
            RETURN 
                COALESCE(n.name, n.id, n.title) as entity_name,
                labels(n) as labels,
                properties(n) as props
            ORDER BY entity_name
            """
            
            result = session.run(query)
            seen_names = set()
            
            for record in result:
                entity_name = record["entity_name"]
                
                # Skip duplicates
                if not entity_name or entity_name in seen_names:
                    continue
                
                seen_names.add(entity_name)
                labels = record["labels"]
                props = record["props"]
                
                # Build rich text representation for embedding
                # Format: "Entity Name | Description/Text | Type: Label1, Label2"
                text_parts = [entity_name]
                
                # Add description or text if available
                if props.get("description"):
                    desc = props["description"][:500]  # Limit to 500 chars
                    text_parts.append(desc)
                elif props.get("text"):
                    text = props["text"][:500]
                    text_parts.append(text)
                
                # Add type/labels
                if labels:
                    text_parts.append(f"Type: {', '.join(labels)}")
                
                # Join with separator
                embedding_text = " | ".join(text_parts)
                
                # Store
                entity_names.append(entity_name)
                entity_texts.append(embedding_text)
                entity_metadata.append({
                    "name": entity_name,
                    "labels": labels,
                    "properties": props
                })
            
            logger.info(f"Extracted {len(entity_names)} unique entities")
            return entity_names, entity_texts, entity_metadata
            
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        return [], [], []


def generate_embeddings(entity_texts: List[str]) -> List[List[float]]:
    """Generate embeddings for all entity texts using batch processing with rate limiting.
    
    Args:
        entity_texts: List of texts to embed
    
    Returns:
        List of embedding vectors
    """
    logger.info(f"Generating embeddings for {len(entity_texts)} entities...")
    
    all_embeddings = []
    total_batches = (len(entity_texts) + BATCH_SIZE - 1) // BATCH_SIZE
    # NVIDIA NIM limit: 40 TPM (transactions per minute) = 1 request per 1.5s
    DELAY_BETWEEN_BATCHES = 2.0  # 2 seconds between batches (safe for 40 TPM)
    
    for i in range(0, len(entity_texts), BATCH_SIZE):
        batch = entity_texts[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} entities)...")
        
        # Try batch embedding first with retry logic
        batch_embeddings = get_embeddings_batch(batch, input_type="passage")
        
        if batch_embeddings:
            all_embeddings.extend(batch_embeddings)
        else:
            # Fallback to individual embeddings if batch fails
            logger.warning(f"Batch embedding failed, falling back to individual embeddings")
            for text in batch:
                embedding = get_embedding(text, input_type="passage")
                if embedding:
                    all_embeddings.append(embedding)
                    time.sleep(1.5)  # 1.5s delay to respect 40 TPM limit
                else:
                    logger.error(f"Failed to embed: {text[:50]}...")
                    return None
        
        # Progress update
        if batch_num % 10 == 0:
            logger.info(f"Progress: {len(all_embeddings)}/{len(entity_texts)} embeddings generated")
        
        # Rate limiting: delay between batches (except last batch)
        if i + BATCH_SIZE < len(entity_texts):
            logger.info(f"Waiting {DELAY_BETWEEN_BATCHES}s before next batch to avoid rate limits...")
            time.sleep(DELAY_BETWEEN_BATCHES)
    
    logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
    return all_embeddings


def build_faiss_index(embeddings: List[List[float]]) -> faiss.Index:
    """Build FAISS index from embeddings.
    
    Args:
        embeddings: List of embedding vectors
    
    Returns:
        FAISS index
    """
    logger.info("Building FAISS index...")
    
    try:
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        logger.info(f"Embedding dimension: {dimension}")
        logger.info(f"Number of vectors: {embeddings_array.shape[0]}")
        
        # Use L2 distance (can also use Inner Product with IndexFlatIP)
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        
        logger.info(f"FAISS index built successfully with {index.ntotal} vectors")
        return index
        
    except Exception as e:
        logger.error(f"Error building FAISS index: {e}")
        return None


def save_index_and_metadata(index: faiss.Index, 
                            entity_names: List[str], 
                            entity_metadata: List[Dict[str, Any]]):
    """Save FAISS index and entity mappings to disk.
    
    Args:
        index: FAISS index
        entity_names: List of entity names
        entity_metadata: List of metadata dicts
    """
    logger.info("Saving FAISS index and metadata to disk...")
    
    try:
        # Save FAISS index
        faiss.write_index(index, INDEX_PATH)
        logger.info(f"✓ Saved FAISS index to {INDEX_PATH}")
        
        # Save entity names list
        with open(ENTITIES_PATH, 'wb') as f:
            pickle.dump(entity_names, f)
        logger.info(f"✓ Saved entity names to {ENTITIES_PATH}")
        
        # Save entity metadata
        metadata_dict = {name: meta for name, meta in zip(entity_names, entity_metadata)}
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata_dict, f)
        logger.info(f"✓ Saved entity metadata to {METADATA_PATH}")
        
        # Print file sizes
        index_size = os.path.getsize(INDEX_PATH) / 1024 / 1024
        entities_size = os.path.getsize(ENTITIES_PATH) / 1024 / 1024
        metadata_size = os.path.getsize(METADATA_PATH) / 1024 / 1024
        
        logger.info(f"\nFile sizes:")
        logger.info(f"  {INDEX_PATH}: {index_size:.2f} MB")
        logger.info(f"  {ENTITIES_PATH}: {entities_size:.2f} MB")
        logger.info(f"  {METADATA_PATH}: {metadata_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error saving files: {e}")


def verify_index():
    """Verify that the saved index can be loaded correctly."""
    logger.info("\nVerifying saved index...")
    
    try:
        # Load FAISS index
        index = faiss.read_index(INDEX_PATH)
        logger.info(f"✓ FAISS index loaded: {index.ntotal} vectors")
        
        # Load entity names
        with open(ENTITIES_PATH, 'rb') as f:
            entity_names = pickle.load(f)
        logger.info(f"✓ Entity names loaded: {len(entity_names)} names")
        
        # Load metadata
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"✓ Metadata loaded: {len(metadata)} entries")
        
        # Verify consistency
        if index.ntotal == len(entity_names) == len(metadata):
            logger.info("✓ All files are consistent!")
            
            # Show sample entities
            logger.info("\nSample entities:")
            for i in range(min(5, len(entity_names))):
                name = entity_names[i]
                labels = metadata[name].get('labels', [])
                logger.info(f"  {i+1}. {name} ({', '.join(labels)})")
            
            return True
        else:
            logger.error(f"✗ Inconsistency detected!")
            logger.error(f"  Index vectors: {index.ntotal}")
            logger.error(f"  Entity names: {len(entity_names)}")
            logger.error(f"  Metadata entries: {len(metadata)}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        return False


def main():
    """Main function to build embeddings and FAISS index."""
    
    # Load tracker state from createKG.py for a unified build report
    tracker.load_state()

    print("=" * 80)
    print("FAISS Embeddings Builder for Hybrid RAG Chatbot")
    print("=" * 80)
    print()

    start_time = datetime.now()
    
    # Step 1: Extract entities from Memgraph
    print("Step 1: Extracting entities from Memgraph...")
    tracker.start_step("Entity Extraction from KG")
    entity_names, entity_texts, entity_metadata = extract_entities_from_kg()
    tracker.end_step("Entity Extraction from KG")

    if not entity_names:
        logger.error("No entities found! Make sure you've run createKG.py first.")
        return

    print(f"✓ Found {len(entity_names)} entities")
    print()

    # Step 2: Generate embeddings
    print("Step 2: Generating embeddings using NVIDIA NIM...")
    tracker.start_step("Embedding Generation (NIM API)")
    embeddings = generate_embeddings(entity_texts)
    tracker.end_step("Embedding Generation (NIM API)")

    if not embeddings or len(embeddings) != len(entity_names):
        logger.error("Failed to generate embeddings for all entities!")
        return

    print(f"✓ Generated {len(embeddings)} embeddings")
    print()

    # Step 3: Build FAISS index
    print("Step 3: Building FAISS index...")
    tracker.start_step("FAISS Index Build")
    index = build_faiss_index(embeddings)
    tracker.end_step("FAISS Index Build")

    if not index:
        logger.error("Failed to build FAISS index!")
        return

    print(f"✓ Built FAISS index with {index.ntotal} vectors")
    print()

    # Step 4: Save to disk
    print("Step 4: Saving to disk...")
    tracker.start_step("Save to Disk")
    save_index_and_metadata(index, entity_names, entity_metadata)
    tracker.end_step("Save to Disk")
    print()

    # Step 5: Verify
    print("Step 5: Verifying saved files...")
    success = verify_index()
    print()
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("=" * 80)
    if success:
        print("✅ SUCCESS! FAISS index built and saved.")
        print()
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Entities indexed: {len(entity_names)}")
        print()
        print("You can now run chatbot_telegram.py - it will load these files automatically!")
    else:
        print("❌ FAILED! Please check the error messages above.")
    print("=" * 80)
    
    # Write build report
    report = tracker.write_report()
    print()
    print(report)

    # Cleanup
    driver.close()


if __name__ == "__main__":
    main()
