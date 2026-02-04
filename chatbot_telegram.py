"""
Knowledge Graph Chatbot using NVIDIA NIM, Memgraph, and Telegram
- HYBRID RAG Architecture: Vector Search + Graph Expansion + LLM Synthesis
- Telegram bot interface for querying knowledge graph
- FAISS vector index for semantic entity search
- Multi-hop graph expansion from seed entities
- Subgraph ranking and pruning
- LLM-NLU fallback for robustness
- Persistent chat history in Memgraph
- Supports multiple users simultaneously

Install:
    pip install openai neo4j python-telegram-bot pytz numpy faiss-cpu

Setup:
    1. Get a bot token from @BotFather on Telegram
    2. Set TELEGRAM_BOT_TOKEN below
    3. Run: python chatbot_telegram.py
    4. First run will build FAISS index (takes a few minutes)

Commands:
    /start - Start conversation with Kuchiko
    /help - Show help message
    /table - View knowledge graph table of contents
    /random - Discover a random entity from the knowledge graph
    /reset - Clear your chat history
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import pytz
import numpy as np
import faiss
import pickle
from neo4j import GraphDatabase, basic_auth
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --------- Configuration ---------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not NVIDIA_API_KEY:
    logger.error("NVIDIA_API_KEY not set. Please set it in your .env file.")
    exit(1)

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN not set. Please set it in your .env file.")
    exit(1)

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "memgraph")
MEMGRAPH_PASS = os.getenv("MEMGRAPH_PASS", "memgraph")

# Model selection
NIM_MODEL = "deepseek-ai/deepseek-v3.1"
NIM_EMBEDDING_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
NIM_API_BASE = "https://integrate.api.nvidia.com/v1"

# Malaysia timezone
MALAYSIA_TZ = pytz.timezone("Asia/Kuala_Lumpur")

# Training data log files
USER_LOG_FILE = "logs/user_interactions.jsonl"  # Full user interaction logs
CHATBOT_LOG_FILE = "logs/chatbot_training_data.jsonl"  # Simplified query-answer pairs

# Hybrid search parameters
TOP_K_SEEDS = 5  # Number of seed entities from vector search
MAX_HOPS = 2  # Maximum hops for graph expansion
MAX_SUBGRAPH_NODES = 50  # Maximum nodes in pruned subgraph

# Initialize OpenAI client for NVIDIA NIM
nim_client = OpenAI(
    base_url=NIM_API_BASE,
    api_key=NVIDIA_API_KEY
)

# Neo4j driver for Memgraph
driver = GraphDatabase.driver(MEMGRAPH_URI, auth=basic_auth(MEMGRAPH_USER, MEMGRAPH_PASS))

# Store conversation history per user
user_conversations: Dict[int, List[Dict[str, str]]] = defaultdict(list)

# Store recent entities/topics per user for continuity
user_context: Dict[int, Dict[str, Any]] = defaultdict(lambda: {"recent_entities": [], "last_keywords": []})

# Store schema (loaded once at startup)
GRAPH_SCHEMA = None

# FAISS index and entity mapping
FAISS_INDEX = None
ENTITY_LIST = []  # List of entity names corresponding to FAISS index positions
ENTITY_METADATA = {}  # Entity name -> {id, properties, type}

# --------- Hybrid Search: Vector + Graph Functions ---------

def get_embedding(text: str, input_type: str = "query") -> List[float]:
    """Get embedding vector from NVIDIA NIM."""
    try:
        response = nim_client.embeddings.create(
            input=[text],
            model=NIM_EMBEDDING_MODEL,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "NONE"}
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None


def build_entity_embeddings() -> Tuple[List[str], List[List[float]], Dict[str, Any]]:
    """Extract all entities from knowledge graph and generate embeddings."""
    logger.info("Building entity embeddings...")
    entity_names = []
    embeddings = []
    metadata = {}
    
    try:
        # Step 1: Extract entity data WITHOUT holding transaction open
        logger.info("Extracting entity data from knowledge graph...")
        entity_data = []
        
        with driver.session() as session:
            query = """
            MATCH (n)
            WHERE NOT n:User AND NOT n:Message
            AND (n.name IS NOT NULL OR n.id IS NOT NULL OR n.title IS NOT NULL)
            RETURN 
                COALESCE(n.name, n.id, n.title) as entity_name,
                labels(n) as labels,
                properties(n) as props
            LIMIT 10000
            """
            
            result = session.run(query)
            seen_names = set()
            
            for record in result:
                entity_name = record["entity_name"]
                if not entity_name or entity_name in seen_names:
                    continue
                
                seen_names.add(entity_name)
                labels = record["labels"]
                props = record["props"]
                
                text_parts = [entity_name]
                if props.get("description"):
                    text_parts.append(props["description"][:500])
                elif props.get("text"):
                    text_parts.append(props["text"][:500])
                if labels:
                    text_parts.append(f"Type: {', '.join(labels)}")
                
                text_for_embedding = " | ".join(text_parts)
                entity_data.append({
                    "name": entity_name,
                    "text": text_for_embedding,
                    "labels": labels,
                    "props": props
                })
        
        logger.info(f"Extracted {len(entity_data)} entities from KG")
        
        # Step 2: Generate embeddings in batches with rate limiting
        # NVIDIA NIM limit: 40 TPM (transactions per minute) = 1 request per 1.5s
        BATCH_SIZE = 50  # Process 50 entities per batch
        DELAY_BETWEEN_BATCHES = 2.0  # 2 seconds between batches (safe for 40 TPM)
        
        for i in range(0, len(entity_data), BATCH_SIZE):
            batch = entity_data[i:i + BATCH_SIZE]
            batch_num = (i // BATCH_SIZE) + 1
            total_batches = (len(entity_data) + BATCH_SIZE - 1) // BATCH_SIZE
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} entities)...")
            
            # Prepare texts for batch embedding
            batch_texts = [item["text"] for item in batch]
            
            # Try batch embedding with retry logic
            batch_embeddings = None
            for attempt in range(3):  # 3 retry attempts
                try:
                    response = nim_client.embeddings.create(
                        input=batch_texts,
                        model=NIM_EMBEDDING_MODEL,
                        encoding_format="float",
                        extra_body={"input_type": "passage", "truncate": "NONE"}
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    break  # Success, exit retry loop
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                        logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt+1}/3...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Error in batch embedding: {e}")
                        break
            
            # If batch failed, try individual embeddings as fallback
            if not batch_embeddings:
                logger.warning("Batch embedding failed, using individual fallback")
                batch_embeddings = []
                for item in batch:
                    try:
                        embedding = get_embedding(item["text"], input_type="passage")
                        if embedding:
                            batch_embeddings.append(embedding)
                            time.sleep(1.5)  # 1.5s delay to respect 40 TPM limit
                        else:
                            batch_embeddings.append(None)
                    except Exception as e:
                        logger.error(f"Failed to embed {item['name']}: {e}")
                        batch_embeddings.append(None)
            
            # Store successful embeddings
            for item, embedding in zip(batch, batch_embeddings):
                if embedding:
                    entity_names.append(item["name"])
                    embeddings.append(embedding)
                    metadata[item["name"]] = {
                        "labels": item["labels"],
                        "properties": item["props"]
                    }
            
            logger.info(f"Progress: {len(embeddings)}/{len(entity_data)} embeddings generated")
            
            # Rate limiting: delay between batches (except last batch)
            if i + BATCH_SIZE < len(entity_data):
                time.sleep(DELAY_BETWEEN_BATCHES)
        
        logger.info(f"Total entities embedded: {len(embeddings)}")
        return entity_names, embeddings, metadata
        
    except Exception as e:
        logger.error(f"Error building entity embeddings: {e}")
        return [], [], {}


def create_faiss_index(embeddings: List[List[float]]) -> faiss.Index:
    """Create FAISS index from embeddings."""
    try:
        embeddings_array = np.array(embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        logger.info(f"Created FAISS index with {index.ntotal} vectors, dimension {dimension}")
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        return None


def save_faiss_index(index: faiss.Index, entity_names: List[str], metadata: Dict[str, Any]):
    """Save FAISS index and entity mappings to disk."""
    try:
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        faiss.write_index(index, os.path.join(data_dir, "faiss_index.bin"))
        with open(os.path.join(data_dir, "entities.pkl"), 'wb') as f:
            pickle.dump(entity_names, f)
        with open(os.path.join(data_dir, "entity_metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        logger.info("Saved FAISS index and mappings")
    except Exception as e:
        logger.error(f"Error saving FAISS index: {e}")


def load_faiss_index() -> Tuple[faiss.Index, List[str], Dict[str, Any]]:
    """Load FAISS index and entity mappings from disk."""
    try:
        data_dir = "data"
        index_path = os.path.join(data_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            return None, [], {}
        index = faiss.read_index(index_path)
        with open(os.path.join(data_dir, "entities.pkl"), 'rb') as f:
            entity_names = pickle.load(f)
        with open(os.path.join(data_dir, "entity_metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        return index, entity_names, metadata
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        return None, [], {}


def vector_search_seeds(query: str, top_k: int = TOP_K_SEEDS) -> List[str]:
    """Search for top-K seed entities using vector similarity."""
    global FAISS_INDEX, ENTITY_LIST
    
    if FAISS_INDEX is None or not ENTITY_LIST:
        logger.warning("FAISS index not initialized")
        return []
    
    try:
        query_embedding = get_embedding(query, input_type="query")
        if not query_embedding:
            return []
        
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = FAISS_INDEX.search(query_vector, top_k)
        seed_entities = [ENTITY_LIST[idx] for idx in indices[0] if idx < len(ENTITY_LIST)]
        
        logger.info(f"Vector search found seeds: {seed_entities}")
        logger.info(f"Distances: {distances[0]}")
        return seed_entities
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []


def expand_from_seeds(seed_entities: List[str], max_hops: int = MAX_HOPS) -> Dict[str, Any]:
    """Expand graph from seed entities using multi-hop traversal."""
    if not seed_entities:
        return {"nodes": {}, "relationships": []}
    
    try:
        with driver.session() as session:
            seed_conditions = []
            for seed in seed_entities:
                seed_lower = seed.lower()
                seed_conditions.append(f"""
                    (seed.name IS NOT NULL AND toLower(seed.name) = '{seed_lower}') OR
                    (seed.id IS NOT NULL AND toLower(seed.id) = '{seed_lower}') OR
                    (seed.title IS NOT NULL AND toLower(seed.title) = '{seed_lower}')
                """)
            
            seed_where = " OR ".join([f"({cond})" for cond in seed_conditions])
            
            query = f"""
            MATCH path = (seed)-[*1..{max_hops}]-(connected)
            WHERE ({seed_where})
            AND NOT seed:User AND NOT seed:Message
            AND NOT connected:User AND NOT connected:Message
            WITH seed, connected, path, size(relationships(path)) as distance, relationships(path) as rels
            LIMIT 200
            RETURN DISTINCT seed, connected, distance, rels
            """
            
            result = session.run(query)
            nodes = {}
            relationships = []
            
            for record in result:
                seed_node = dict(record["seed"])
                connected_node = dict(record["connected"])
                distance = record["distance"]
                rels = record["rels"]
                
                # Extract relationship types in Python (Memgraph doesn't support list comprehension in Cypher)
                rel_types = [rel.type for rel in rels] if rels else []
                
                seed_id = seed_node.get("name") or seed_node.get("id") or seed_node.get("title")
                connected_id = connected_node.get("name") or connected_node.get("id") or connected_node.get("title")
                
                if seed_id:
                    nodes[seed_id] = {"properties": seed_node, "distance": 0, "is_seed": True}
                if connected_id:
                    if connected_id not in nodes:
                        nodes[connected_id] = {"properties": connected_node, "distance": distance, "is_seed": False}
                    else:
                        nodes[connected_id]["distance"] = min(nodes[connected_id]["distance"], distance)
                
                if seed_id and connected_id:
                    relationships.append({"source": seed_id, "target": connected_id, "types": rel_types, "distance": distance})
            
            logger.info(f"Expanded to {len(nodes)} nodes and {len(relationships)} relationships")
            return {"nodes": nodes, "relationships": relationships}
    except Exception as e:
        logger.error(f"Error expanding from seeds: {e}")
        return {"nodes": {}, "relationships": []}


def rank_and_prune_subgraph(subgraph: Dict[str, Any], query: str, max_nodes: int = MAX_SUBGRAPH_NODES) -> Dict[str, Any]:
    """Rank and prune subgraph based on relevance scores."""
    nodes = subgraph.get("nodes", {})
    relationships = subgraph.get("relationships", [])
    
    if not nodes:
        return subgraph
    
    node_scores = {}
    for node_id, node_data in nodes.items():
        score = 0.0
        distance = node_data.get("distance", 99)
        if node_data.get("is_seed"):
            score += 100
        else:
            score += max(0, 50 - (distance * 10))
        
        props = node_data.get("properties", {})
        if props.get("description") or props.get("text"):
            score += 20
        
        degree = sum(1 for rel in relationships if rel["source"] == node_id or rel["target"] == node_id)
        score += min(degree * 2, 30)
        node_scores[node_id] = score
    
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
    top_node_ids = set([node_id for node_id, _ in sorted_nodes[:max_nodes]])
    
    pruned_nodes = {node_id: nodes[node_id] for node_id in top_node_ids}
    pruned_relationships = [rel for rel in relationships if rel["source"] in top_node_ids and rel["target"] in top_node_ids]
    
    logger.info(f"Pruned subgraph to {len(pruned_nodes)} nodes and {len(pruned_relationships)} relationships")
    return {"nodes": pruned_nodes, "relationships": pruned_relationships}


# --------- Chat History Persistence Functions ---------

def init_chat_schema():
    """Initialize chat history schema in Memgraph if not exists."""
    try:
        with driver.session() as session:
            # Create indices for faster lookups
            session.run("CREATE INDEX ON :User(user_id);")
            session.run("CREATE INDEX ON :Message(timestamp);")
            logger.info("Chat history schema initialized")
    except Exception as e:
        logger.info(f"Schema already exists or error: {e}")


def save_message_to_graph(user_id: int, role: str, content: str, entities: List[str] = None, user_info: Dict[str, str] = None):
    """Save a chat message to Memgraph with Malaysia timezone.
    
    Structure:
    (User {user_id, username, first_name, last_name, created_at})
    -[:SENT {timestamp}]->
    (Message {role, content, entities, timestamp})
    """
    try:
        with driver.session() as session:
            # Get current time in Malaysia timezone
            timestamp = datetime.now(MALAYSIA_TZ).isoformat()
            
            # Prepare user info
            username = user_info.get('username', '') if user_info else ''
            first_name = user_info.get('first_name', '') if user_info else ''
            last_name = user_info.get('last_name', '') if user_info else ''
            
            query = """
            MERGE (u:User {user_id: $user_id})
            ON CREATE SET u.created_at = $timestamp,
                          u.username = $username,
                          u.first_name = $first_name,
                          u.last_name = $last_name
            ON MATCH SET u.username = $username,
                         u.first_name = $first_name,
                         u.last_name = $last_name
            CREATE (m:Message {
                role: $role,
                content: $content,
                entities: $entities,
                timestamp: $timestamp
            })
            CREATE (u)-[:SENT {timestamp: $timestamp}]->(m)
            """
            
            session.run(query, 
                       user_id=user_id,
                       username=username,
                       first_name=first_name,
                       last_name=last_name,
                       role=role, 
                       content=content,
                       entities=entities or [],
                       timestamp=timestamp)
            
            logger.info(f"Saved {role} message for user {user_id} ({first_name})")
    except Exception as e:
        logger.error(f"Error saving message to graph: {e}")


def get_chat_history_from_graph(user_id: int, limit_pairs: int = 10) -> List[Dict[str, str]]:
    """Retrieve chat history from Memgraph for a specific user.
    
    Returns last N Q&A pairs (limit_pairs * 2 messages) in chronological order,
    based on Malaysia timezone."""
    try:
        with driver.session() as session:
            # Get last N*2 messages (N pairs of user+assistant messages)
            query = """
            MATCH (u:User {user_id: $user_id})-[s:SENT]->(m:Message)
            RETURN m.role as role, m.content as content, m.timestamp as timestamp
            ORDER BY m.timestamp DESC
            LIMIT $limit
            """
            
            result = session.run(query, user_id=user_id, limit=limit_pairs * 2)
            messages = []
            
            for record in result:
                messages.append({
                    "role": record["role"],
                    "content": record["content"]
                })
            
            # Reverse to get chronological order
            messages.reverse()
            
            logger.info(f"Retrieved {len(messages)} messages ({len(messages)//2} Q&A pairs) for user {user_id}")
            return messages
            
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return []


def get_recent_entities_from_graph(user_id: int, limit: int = 20) -> List[str]:
    """Get recently discussed entities for a user from their chat history."""
    try:
        with driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[:SENT]->(m:Message)
            WHERE m.entities IS NOT NULL AND size(m.entities) > 0
            RETURN m.entities as entities, m.timestamp as timestamp
            ORDER BY m.timestamp DESC
            LIMIT 10
            """
            
            result = session.run(query, user_id=user_id)
            all_entities = []
            
            for record in result:
                entities = record["entities"]
                if entities:
                    all_entities.extend(entities)
            
            # Return unique entities, preserving recent order
            seen = set()
            unique_entities = []
            for entity in all_entities:
                if entity not in seen:
                    seen.add(entity)
                    unique_entities.append(entity)
                    if len(unique_entities) >= limit:
                        break
            
            return unique_entities
            
    except Exception as e:
        logger.error(f"Error retrieving recent entities: {e}")
        return []


def clear_user_chat_history(user_id: int):
    """Clear all chat history for a specific user."""
    try:
        with driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[:SENT]->(m:Message)
            DETACH DELETE m
            """
            session.run(query, user_id=user_id)
            logger.info(f"Cleared chat history for user {user_id}")
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")


# --------- Knowledge Graph Query Functions ---------

def get_graph_schema() -> str:
    """Retrieve the schema of the knowledge graph by discovering it from actual data."""
    with driver.session() as session:
        schema_info = {
            "node_labels": [],
            "relationship_types": [],
            "sample_properties": {},
            "node_count": 0,
            "relationship_count": 0
        }
        
        try:
            # Get node count
            count_result = session.run("MATCH (n) RETURN count(n) AS cnt")
            node_count = count_result.single()["cnt"]
            schema_info["node_count"] = node_count
            
            # Get relationship count
            rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            rel_count = rel_count_result.single()["cnt"]
            schema_info["relationship_count"] = rel_count
            
            # Discover node labels and their properties
            labels_result = session.run("""
                MATCH (n)
                RETURN distinct labels(n) AS labels, count(n) AS cnt
                ORDER BY cnt DESC
            """)
            
            for record in labels_result:
                node_labels = record["labels"]
                if node_labels:
                    for label in node_labels:
                        if label not in schema_info["node_labels"]:
                            schema_info["node_labels"].append(label)
                    
                    # Get sample properties for this label combination
                    label_str = ":" + node_labels[0]
                    try:
                        sample_result = session.run(f"MATCH (n{label_str}) RETURN properties(n) AS props LIMIT 1")
                        sample_record = sample_result.single()
                        if sample_record and sample_record["props"]:
                            schema_info["sample_properties"][node_labels[0]] = list(sample_record["props"].keys())
                    except:
                        pass
            
            # Discover relationship types
            rel_types_result = session.run("""
                MATCH ()-[r]->()
                RETURN distinct type(r) AS relType, count(r) AS cnt
                ORDER BY cnt DESC
            """)
            
            for record in rel_types_result:
                rel_type = record["relType"]
                if rel_type:
                    schema_info["relationship_types"].append(rel_type)
            
        except Exception as e:
            logger.warning(f"Error discovering schema: {e}")
            schema_info["warning"] = str(e)
        
        return json.dumps(schema_info, indent=2)


def execute_cypher_query(cypher: str) -> List[Dict[str, Any]]:
    """Execute a Cypher query and return results."""
    try:
        with driver.session() as session:
            result = session.run(cypher)
            records = []
            for record in result:
                records.append(dict(record))
            return records
    except Exception as e:
        logger.error(f"Error executing Cypher query: {e}")
        return [{"error": str(e)}]


def extract_keywords(question: str, schema: str, previous_entities: List[str] = None) -> Dict[str, Any]:
    """Use LLM as NLU to extract keywords and query intent from the user's question.
    
    Includes previous conversation entities to understand follow-up questions."""
    
    context_info = ""
    if previous_entities:
        context_info = f"\n\nPrevious conversation context (recently discussed entities):\n{', '.join(previous_entities[:5])}"
    
    nlu_prompt = f"""You are a Natural Language Understanding (NLU) expert. Analyze the user's question and extract keywords that would be useful for querying a knowledge graph.

Knowledge Graph Schema:
{schema}{context_info}

User Question: {question}

Extract and return ONLY a JSON object (no other text) with the following structure:
{{
    "keywords": ["keyword1", "keyword2", "keyword1_singular", "keyword1_plural"],
    "entity_types": ["type1", "type2"],
    "query_intent": "search|relationship|property|comparison|list",
    "confidence": 0.0-1.0
}}

IMPORTANT INSTRUCTIONS FOR KEYWORDS:
- For each main keyword, include BOTH singular and plural forms
- Examples: if you extract "Brookes", also add "Brooke"
- If you extract "miners", also add "miner"
- This helps with database searching since data might use either form
- If the question seems like a follow-up (e.g., just an entity name), include previous context entities as additional keywords
- Avoid duplicates

Where:
- keywords: Main terms/entities to search for (including singular + plural variants)
- entity_types: Expected node labels (Person, Organization, Location, etc.)
- query_intent: What the user is trying to do
- confidence: How confident you are in the extraction (0-1)"""

    try:
        response = nim_client.chat.completions.create(
            model=NIM_MODEL,
            messages=[
                {"role": "system", "content": "You are an NLU expert. Return ONLY valid JSON, no explanations."},
                {"role": "user", "content": nlu_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Clean up if wrapped in markdown
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        keywords_data = json.loads(response_text)
        logger.info(f"Extracted keywords: {keywords_data}")
        return keywords_data
    
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return {"keywords": [], "entity_types": [], "query_intent": "search", "confidence": 0.0}


def build_predefined_cypher_query(keywords: List[str], entity_types: List[str], query_intent: str) -> str:
    """Build PRIMARY Cypher query for factual information about the main entity.
    
    This query returns direct matches without extensive neighbor information.
    Searches all keyword variants (singular/plural) for better coverage."""
    
    if not keywords:
        return "MATCH (n) RETURN n LIMIT 5"
    
    # Build search conditions for all keywords (including singular/plural variants)
    # Create WHERE clauses that search across all keywords
    keyword_conditions = []
    for kw in keywords:
        kw_lower = kw.lower()
        keyword_conditions.append(f"""(
                (n.name IS NOT NULL AND toLower(n.name) CONTAINS '{kw_lower}') OR
                (n.id IS NOT NULL AND toLower(n.id) CONTAINS '{kw_lower}') OR
                (n.title IS NOT NULL AND toLower(n.title) CONTAINS '{kw_lower}') OR
                (n.description IS NOT NULL AND toLower(n.description) CONTAINS '{kw_lower}') OR
                (n.text IS NOT NULL AND toLower(n.text) CONTAINS '{kw_lower}') OR
                (n.label IS NOT NULL AND toLower(n.label) CONTAINS '{kw_lower}')
            )""")
    
    keyword_where = " OR ".join(keyword_conditions)
    
    # PRIMARY query templates - focused on the main entity
    templates = {
        "search": f"""
            MATCH (n)
            WHERE {keyword_where}
            RETURN n LIMIT 10
        """,
        
        "relationship": f"""
            MATCH (n1)-[r]-(n2)
            WHERE {keyword_where}
            RETURN {{
                source: n1,
                relation: type(r),
                target: n2
            }} LIMIT 10
        """,
        
        "property": f"""
            MATCH (n)
            WHERE {keyword_where}
            RETURN n LIMIT 10
        """,
        
        "list": f"""
            MATCH (n)
            WHERE {keyword_where}
            RETURN n LIMIT 15
        """
    }
    
    query = templates.get(query_intent, templates["search"])
    return query.strip()


def build_secondary_exploration_query(keywords: List[str]) -> str:
    """Build SECONDARY Cypher query to get neighboring entities for exploration.
    
    This query finds connected entities with at least basic information."""
    
    if not keywords:
        return """MATCH (n)-[r]-(neighbor) 
                  WHERE neighbor.name IS NOT NULL OR neighbor.id IS NOT NULL
                  RETURN DISTINCT neighbor, type(r) as relation_type
                  LIMIT 10"""
    
    # Build search conditions
    keyword_conditions = []
    for kw in keywords:
        kw_lower = kw.lower()
        keyword_conditions.append(f"""(
                (n.name IS NOT NULL AND toLower(n.name) CONTAINS '{kw_lower}') OR
                (n.id IS NOT NULL AND toLower(n.id) CONTAINS '{kw_lower}') OR
                (n.title IS NOT NULL AND toLower(n.title) CONTAINS '{kw_lower}') OR
                (n.description IS NOT NULL AND toLower(n.description) CONTAINS '{kw_lower}') OR
                (n.text IS NOT NULL AND toLower(n.text) CONTAINS '{kw_lower}') OR
                (n.label IS NOT NULL AND toLower(n.label) CONTAINS '{kw_lower}')
            )""")
    
    keyword_where = " OR ".join(keyword_conditions)
    
    # Get neighboring entities with at least a name or ID
    # Return all properties so LLM can judge which have enough info
    query = f"""
        MATCH (n)-[r]-(neighbor)
        WHERE {keyword_where}
        AND (neighbor.name IS NOT NULL OR neighbor.id IS NOT NULL OR neighbor.title IS NOT NULL)
        RETURN DISTINCT neighbor, type(r) as relation_type
        LIMIT 15
    """
    
    return query.strip()


def get_context_from_kg(question: str, schema: str, previous_entities: List[str] = None) -> Dict[str, Any]:
    """Generate context using two-query approach: PRIMARY for facts, SECONDARY for exploration.
    
    Process:
    1. Use LLM as NLU to extract keywords and query intent (with previous entities)
    2. Build and execute PRIMARY query for factual information
    3. Build and execute SECONDARY query for 2-hop exploration candidates
    4. Return both results separately with extracted keywords"""
    
    try:
        # Step 1: Extract keywords and intent using NLU with previous entities
        logger.info(f"Extracting keywords from: {question}")
        if previous_entities:
            logger.info(f"Using previous context: {previous_entities[:3]}")
        keywords_data = extract_keywords(question, schema, previous_entities)
        
        keywords = keywords_data.get("keywords", [])
        entity_types = keywords_data.get("entity_types", [])
        query_intent = keywords_data.get("query_intent", "search")
        confidence = keywords_data.get("confidence", 0.0)
        
        logger.info(f"Intent: {query_intent}, Confidence: {confidence}")
        
        # Step 2: Build and execute PRIMARY query for factual answer
        primary_query = build_predefined_cypher_query(keywords, entity_types, query_intent)
        logger.info(f"Executing PRIMARY query")
        primary_results = execute_cypher_query(primary_query)
        
        if not primary_results or (len(primary_results) == 1 and "error" in primary_results[0]):
            logger.warning(f"No primary results found for keywords: {keywords}")
            return {
                "primary": {"info": "No results found", "keywords_tried": keywords},
                "secondary": [],
                "keywords": keywords
            }
        
        logger.info(f"Retrieved {len(primary_results)} primary results")
        
        # Step 3: Build and execute SECONDARY query for exploration candidates
        secondary_query = build_secondary_exploration_query(keywords)
        logger.info(f"Executing SECONDARY query for 2-hop neighbors")
        secondary_results = execute_cypher_query(secondary_query)
        
        if not secondary_results or (len(secondary_results) == 1 and "error" in secondary_results[0]):
            logger.warning(f"No secondary results found")
            secondary_results = []
        else:
            logger.info(f"Retrieved {len(secondary_results)} secondary results")
        
        # Step 4: Return both results with truncation if needed
        primary_json = json.dumps(primary_results, indent=2, default=str)
        secondary_json = json.dumps(secondary_results, indent=2, default=str)
        
        MAX_CHARS = 50000
        if len(primary_json) > MAX_CHARS:
            primary_json = primary_json[:MAX_CHARS] + "\n... (truncated)"
        if len(secondary_json) > MAX_CHARS:
            secondary_json = secondary_json[:MAX_CHARS] + "\n... (truncated)"
        
        return {
            "primary": json.loads(primary_json) if len(primary_json) < MAX_CHARS else primary_results[:5],
            "secondary": json.loads(secondary_json) if len(secondary_json) < MAX_CHARS else secondary_results[:10],
            "keywords": keywords
        }
    
    except Exception as e:
        logger.error(f"Error in two-query pipeline: {e}")
        return {"primary": {"error": str(e)}, "secondary": [], "keywords": []}


def classify_user_intent(question: str, previous_entities: List[str] = None) -> Dict[str, Any]:
    """Use LLM to classify the user's intent.
    
    Returns:
        {
            "intent": "greeting" | "small_talk" | "follow_up" | "knowledge_query" | "meta",
            "needs_kg": bool,
            "needs_context": bool
        }
    """
    
    context_hint = ", ".join(previous_entities[:5]) if previous_entities else "None"
    
    prompt = f"""Classify the user's intent for this query.

Recent conversation context: {context_hint}

User query: "{question}"

Intent categories:
1. GREETING - Simple greetings, thanks, goodbyes (hi, hello, thanks, bye)
2. SMALL_TALK - Casual conversation not about knowledge base (how are you, what's up)
3. FOLLOW_UP - Refers to previous context with pronouns or asks for more (tell me more, what about him)
4. KNOWLEDGE_QUERY - Asks about specific information from knowledge base (who is X, what is Y)
5. META - Asks about bot capabilities (what can you do, how do you work)

Respond with ONLY the category name: GREETING, SMALL_TALK, FOLLOW_UP, KNOWLEDGE_QUERY, or META"""

    try:
        response = nim_client.chat.completions.create(
            model=NIM_MODEL,
            messages=[
                {"role": "system", "content": "You are an intent classifier. Answer with only one word."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        intent = response.choices[0].message.content.strip().upper()
        
        # Map intent to needs
        intent_map = {
            "GREETING": {"intent": "greeting", "needs_kg": False, "needs_context": False},
            "SMALL_TALK": {"intent": "small_talk", "needs_kg": False, "needs_context": False},
            "FOLLOW_UP": {"intent": "follow_up", "needs_kg": True, "needs_context": True},
            "KNOWLEDGE_QUERY": {"intent": "knowledge_query", "needs_kg": True, "needs_context": False},
            "META": {"intent": "meta", "needs_kg": False, "needs_context": False}
        }
        
        result = intent_map.get(intent, {"intent": "knowledge_query", "needs_kg": True, "needs_context": False})
        logger.info(f"[INTENT] Classified '{question}' as: {result['intent']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        # Fallback: assume knowledge query
        return {"intent": "knowledge_query", "needs_kg": True, "needs_context": False}


def handle_casual_response(question: str, intent_type: str) -> str:
    """Generate casual response for non-KG queries (greetings, small talk)."""
    
    responses = {
        "greeting": [
            "Hey there! 👋 I'm Kuchiko, your knowledge guide! What would you like to explore today?",
            "Hello! 🌿 Great to see you! Ask me anything about the knowledge base.",
            "Hi! 😊 Ready to dive into some interesting topics? What catches your curiosity?",
        ],
        "small_talk": [
            "I'm doing great, thanks for asking! I'm always excited to share knowledge. What would you like to learn about?",
            "I'm here and ready to help! 🌿 Got any questions about the knowledge base?",
            "All good here! I love chatting about history, places, and people. What interests you?",
        ],
        "meta": [
            "I'm Kuchiko, a knowledge graph chatbot! I can answer questions about entities, relationships, and information in my knowledge base. I remember our conversation context too, so you can ask follow-up questions naturally. What would you like to explore?",
            "I help you explore information from a knowledge graph! Ask me about people, places, events, or relationships. I can also suggest related topics to dive deeper into. Try asking about something specific!",
        ]
    }
    
    import random
    response_list = responses.get(intent_type, responses["greeting"])
    return random.choice(response_list)


def get_context_hybrid(question: str, previous_entities: List[str] = None, needs_context: bool = False) -> Dict[str, Any]:
    """HYBRID approach: Vector search + Graph expansion + Ranking.
    
    Pipeline:
    1. Embed query (enriched with context if needs_context is True)
    2. Vector search for top-K seed entities
    3. Multi-hop graph expansion from seeds
    4. Rank and prune subgraph
    5. Return pruned subgraph for LLM synthesis
    
    Falls back to LLM-NLU if vector search fails.
    """
    global FAISS_INDEX
    
    try:
        # Enrich query with context if it's a follow-up
        query_for_search = question
        if needs_context and previous_entities:
            context_hint = " ".join(previous_entities[:3])
            query_for_search = f"{question} {context_hint}"
            logger.info(f"[HYBRID] Context-enriched query: {query_for_search}")
        
        # Step 1 & 2: Vector search for seed entities
        logger.info(f"[HYBRID] Starting vector search for: {query_for_search}")
        seed_entities = vector_search_seeds(query_for_search, top_k=TOP_K_SEEDS)
        
        # Boost with recent entities if context was needed
        if needs_context and previous_entities:
            # Add top recent entities as bonus seeds (avoid duplicates)
            for entity in previous_entities[:3]:
                if entity not in seed_entities:
                    seed_entities.append(entity)
            logger.info(f"[HYBRID] Added context seeds: {previous_entities[:3]}")
        
        # Fallback to LLM-NLU if vector search fails
        if not seed_entities:
            logger.warning("[HYBRID] Vector search failed, falling back to LLM-NLU")
            return get_context_from_kg(question, GRAPH_SCHEMA, previous_entities)
        
        # Step 3: Expand graph from seed entities
        logger.info(f"[HYBRID] Expanding graph from {len(seed_entities)} seeds")
        subgraph = expand_from_seeds(seed_entities, max_hops=MAX_HOPS)
        
        if not subgraph.get("nodes"):
            logger.warning("[HYBRID] No graph expansion, falling back to LLM-NLU")
            return get_context_from_kg(question, GRAPH_SCHEMA, previous_entities)
        
        # Step 4: Rank and prune subgraph
        logger.info(f"[HYBRID] Ranking and pruning subgraph")
        pruned_subgraph = rank_and_prune_subgraph(subgraph, question, max_nodes=MAX_SUBGRAPH_NODES)
        
        # Format for LLM
        return {
            "method": "hybrid",
            "seeds": seed_entities,
            "subgraph": pruned_subgraph,
            "keywords": seed_entities
        }
        
    except Exception as e:
        logger.error(f"[HYBRID] Error in hybrid pipeline: {e}")
        # Fallback to LLM-NLU
        return get_context_from_kg(question, GRAPH_SCHEMA, previous_entities)


def chat_with_kg(question: str, schema: str, conversation_history: List[Dict[str, str]] = None, previous_entities: List[str] = None) -> tuple[str, List[str]]:
    """Main chatbot function using HYBRID RAG approach with intent classification.
    
    Classifies user intent first, then:
    - Greetings/small talk: Direct casual response
    - Follow-up: Uses vector search + graph expansion with context enrichment
    - Knowledge query: Uses vector search + graph expansion
    - Falls back to LLM-NLU if needed
    
    Returns: (answer, list of entities discussed for future context)"""
    
    if conversation_history is None:
        conversation_history = []
    
    logger.info(f"User question: {question}")
    
    # Step 1: Classify user intent
    intent_data = classify_user_intent(question, previous_entities)
    intent = intent_data["intent"]
    needs_kg = intent_data["needs_kg"]
    needs_context = intent_data["needs_context"]
    
    logger.info(f"[INTENT] {intent} | needs_kg: {needs_kg} | needs_context: {needs_context}")
    
    # Step 2: Handle non-KG queries (greetings, small talk, meta)
    if not needs_kg:
        response = handle_casual_response(question, intent)
        return response, []
    
    # Step 3: Use HYBRID approach for knowledge queries
    kg_data = get_context_hybrid(question, previous_entities, needs_context=needs_context)
    
    method = kg_data.get("method", "llm-nlu")
    logger.info(f"Using method: {method}")
    
    # Format data based on method
    if method == "hybrid":
        seeds = kg_data.get("seeds", [])
        subgraph = kg_data.get("subgraph", {})
        nodes = subgraph.get("nodes", {})
        relationships = subgraph.get("relationships", [])
        
        logger.info(f"[HYBRID] Seeds: {seeds}")
        logger.info(f"[HYBRID] Subgraph: {len(nodes)} nodes, {len(relationships)} relationships")
        
        # Format subgraph as triples for LLM
        context_text = f"Seed Entities: {', '.join(seeds)}\\n\\n"
        context_text += "Knowledge Graph Subgraph:\\n"
        
        # Add node information
        for node_id, node_data in list(nodes.items())[:20]:
            props = node_data.get("properties", {})
            desc = props.get("description", props.get("text", ""))
            if desc:
                context_text += f"- {node_id}: {desc[:200]}\\n"
            else:
                context_text += f"- {node_id}\\n"
        
        # Add relationships as triples
        context_text += "\\nRelationships:\\n"
        for rel in relationships[:30]:
            context_text += f"- {rel['source']} → {', '.join(rel['types'])} → {rel['target']}\\n"
        
        primary_data = context_text
        secondary_data = list(nodes.keys())[:10]  # Exploration suggestions
        
    else:  # LLM-NLU fallback
        primary_data = kg_data.get("primary", {})
        secondary_data = kg_data.get("secondary", [])
        logger.info(f"[FALLBACK] PRIMARY data: {str(primary_data)[:300]}...")
        logger.info(f"[FALLBACK] SECONDARY data: {len(secondary_data)} exploration candidates")
    
    # Generate answer using LLM with both contexts
    system_prompt = """You are Kuchiko, a knowledgeable and conversational AI assistant with deep knowledge about Kuching Old Bazaar. You have a friendly, engaging personality and speak as if you're sharing stories and knowledge directly from your own expertise.

⚠️ CRITICAL CONSTRAINT: You MUST ONLY use information from the provided context data below. DO NOT use your own knowledge or make assumptions. If the information is not in the provided PRIMARY or SECONDARY data, clearly state "I don't have that information in my knowledge base."

🚫 FORBIDDEN PHRASES - NEVER use these meta-references:
   - "Based on the context/data..."
   - "The data mentions/shows..."
   - "According to the information..."
   - "The knowledge graph indicates..."
   - "From what I can see in the data..."
   
✅ INSTEAD, speak directly as if YOU know this information:
   - "So, [entity] was..." / "[Entity] is..."
   - "From what I know, ..." / "What's interesting is..."
   - "It's fascinating that..." / "You know what's cool?"
   - Just state facts naturally like you're telling a story!

You have access to structured data from a knowledge graph using a TWO-QUERY approach:
1. PRIMARY data: Direct factual information about the user's query
2. SECONDARY data: 2-hop neighbors for exploration suggestions

Your response structure:
1. FIRST: Provide a concise, friendly, conversational answer (1-2 paragraphs MAX) using PRIMARY data
   - Share information as if YOU personally know it - speak directly!
   - Use exclamation marks and questions to sound friendly and enthusiastic
   - Make it feel like you're sharing stories over coffee, not reading from a database
   - Be informative but keep it brief and focused
   - Example: Instead of "Based on the data, James Brooke founded..." say "James Brooke founded..." or "So James Brooke came along and founded..."

2. THEN: Add exploration suggestions from SECONDARY data (after your main answer)
   - Check the SECONDARY data - look at the neighbor entities returned
   - INTELLIGENT FILTERING: Only suggest entities that have enough information:
     * Has a 'description' or 'text' field, OR
     * Has multiple meaningful properties (not just 'name' and 'id'), OR
     * Has properties like 'title', 'label' with actual content
   - Skip entities that are just bare references (only have a name/id with no other info)
   - Extract actual entity names from entities that pass this check
   - Leave a blank line before suggestions
   - Vary your suggestion format! Use different phrasings like:
     * "It connects to..." / "This links to..." / "Related to..."
     * "Curious about...?" / "Want to explore...?" / "Interested in learning more about..."
     * "Which one do you want to explore?" / "What catches your interest?" / "Where should we dive next?"
   - Format with bullet points (•)
   - Keep suggestions SHORT - just the exact entity names from the data
   - NO markdown formatting (no **, no _, no special formatting)
   - Keep it clean and simple (3-5 items max)
   - Example: "• James Brooke" or "• Sarawak River" (entities with actual content)
   - IMPORTANT: If no entities have enough information, skip suggestions entirely!

Your communication style:
1. Be warm, enthusiastic, and naturally conversational
2. Share information as if it's YOUR knowledge - speak directly, not about "data"
3. Tell stories and make connections interesting, but stay concise
4. Use varied, friendly language - be expressive!
5. Keep responses SHORT and to the point (1-2 paragraphs for main answer)

Remember: You're Kuchiko, a friendly knowledgeable companion who KNOWS these things, not a search engine reading data. Share your knowledge naturally as if telling a friend!"""

    # Format user prompt based on data type
    if method == "hybrid":
        user_prompt = f"""Knowledge Graph Context (from vector search + graph expansion):

{primary_data}

User Question: {question}

REMINDER: Answer ONLY using the knowledge graph context provided above. Do not use external knowledge. If the information isn't in the context, say so.

Provide your answer as Kuchiko with exploration suggestions:"""
    else:
        user_prompt = f"""PRIMARY Data (for answering the question):
{json.dumps(primary_data, indent=2, default=str)}

SECONDARY Data (for exploration suggestions):
{json.dumps(secondary_data[:15], indent=2, default=str) if secondary_data else '[]'}

User Question: {question}

REMINDER: Answer ONLY using the PRIMARY and SECONDARY data provided above. Do not use external knowledge. If the information isn't in the data, say so.

Provide your answer as Kuchiko with exploration suggestions:"""

    try:
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        for msg in conversation_history[-6:]:  # Keep last 6 messages for context
            messages.append(msg)
        
        messages.append({"role": "user", "content": user_prompt})
        
        response = nim_client.chat.completions.create(
            model=NIM_MODEL,
            messages=messages,
            temperature=0.8,  # Increased for more varied responses
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract entities discussed for future context
        discussed_entities = kg_data.get("keywords", [])
        
        # For hybrid method, also add seed entities and node names
        if method == "hybrid":
            discussed_entities.extend(kg_data.get("seeds", []))
            subgraph = kg_data.get("subgraph", {})
            nodes = subgraph.get("nodes", {})
            discussed_entities.extend(list(nodes.keys())[:10])
        else:
            # Extract from primary data for LLM-NLU method
            primary_data_obj = kg_data.get("primary", {})
            if isinstance(primary_data_obj, list):
                for item in primary_data_obj:
                    if isinstance(item, dict):
                        for key in ['name', 'id', 'title']:
                            if key in item and item[key]:
                                discussed_entities.append(str(item[key]))
        
        # Remove duplicates and limit
        discussed_entities = list(dict.fromkeys(discussed_entities))[:15]
        
        return answer, discussed_entities
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return f"I encountered an error while processing your question: {str(e)}", []


# --------- Training Data Logging ---------

def log_training_data(user_id: int, user_query: str, bot_answer: str, user_info: Dict[str, str] = None) -> None:
    """Log query-answer pairs in two formats:
    1. Full user interaction log with all metadata
    2. Simplified chatbot training data with just query-answer pairs
    """
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Get timestamp in custom format: DD-MM-YYYY HH:MM
        timestamp = datetime.now(MALAYSIA_TZ).strftime("%d-%m-%Y %H:%M")
        
        # 1. Full user interaction log
        user_log_entry = {
            "timestamp": timestamp,
            "user_id": user_id,
            "query": user_query,
            "answer": bot_answer
        }
        
        # Add user info if available
        if user_info:
            user_log_entry["user_info"] = user_info
        
        # Append to user log file with pretty formatting
        with open(USER_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(user_log_entry, ensure_ascii=False, indent=2))
            f.write('\n\n')  # Add blank line separator between entries
        
        # 2. Simplified chatbot training data (query-answer only)
        chatbot_log_entry = {
            "query": user_query,
            "answer": bot_answer
        }
        
        # Append to chatbot training log file with pretty formatting
        with open(CHATBOT_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(chatbot_log_entry, ensure_ascii=False, indent=2))
            f.write('\n\n')  # Add blank line separator between entries
        
        logger.debug(f"Logged training data for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error logging training data: {e}")


# --------- Telegram Bot Handlers ---------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user_id = update.effective_user.id
    first_name = update.effective_user.first_name or 'friend'
    
    welcome_message = f"""🌿 *Welcome to Kuching Old Bazaar chatbot, {first_name}!* 🌿

My name is Kuchiko, your friendly KOB assistant! I can help you explore and answer any questions about that comes to mind.

*How to use:*
📝 Just type your questions naturally
🔍 Ask about anything related to Kuching Old Bazaar
💬 Have a conversation - I remember our chat context

*Commands:*
/help - Show this help message
/table - Unsure what to ask? View the table of contents!
/random - Discover a random topic from the knowledge graph
/reset - Start a fresh conversation

Let's chat! What would you like to know?"""
    
    await update.message.reply_text(welcome_message, parse_mode='Markdown')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_text = """🌿 *Kuchiko Help* 🌿

*Available Commands:*
/start - Start or restart conversation
/table - View table of contents
/random - Discover a random topic
/reset - Clear conversation history

"""
    
    await update.message.reply_text(help_text, parse_mode='Markdown')


async def table_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /table command - show hard-coded table of contents of KOB knowledge graph."""
    
    # Hard-coded table of contents for Kuching Old Bazaar (no Markdown to avoid parsing issues)
    toc_text = """📚 Kuching Old Bazaar - Table of Contents

🏛️ Historical Sketches
  • The Golden Era of Kuching Old Bazaar
  • Streets and Places with its Interesting Old Stories
  • The Four Institutions of Chinese Society: Temples, Associations, Schools, and Commerce
  • Trade activities at Kuching Old Bazaar
  • The Cathedral on the Hill: The Anglican Mission in Kuching
  • Heritage Buildings around KOB
  • Pioneers of KOB
  • The White Rajah and The Old Bazaar

🌆 Life Changes
  • The Fashion Trends of The Old Bazaar
  • The Old Bazaar's Famous Food
  • Home at Nanyang Veranda
  • Going Places: Transportation of The Old Bazaar
  • What were the Old Pastimes?
  • Traditional Events of KOB 
  • Grassroots Stories of the Backstreets
  • Do you remember these interesting market slang words?
  • Invasion: The Kuching Old Bazaar Under Japanese Occupation
  • Disaster in The Old Bazaar's History

💡 Tip: Ask me about any topic to learn more!"""
    
    await update.message.reply_text(toc_text)


async def random_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /random command - pick a random entity from the knowledge graph and describe it."""
    user_id = update.effective_user.id
    first_name = update.effective_user.first_name or 'friend'
    
    try:
        # Send typing indicator
        await update.message.chat.send_action(action="typing")
        
        with driver.session() as session:
            # Get a random entity from the knowledge graph
            query = """
            MATCH (e:Entity)
            WHERE NOT e:User AND NOT e:Message AND NOT e:Document AND NOT e:Segment
            AND (e.name IS NOT NULL OR e.title IS NOT NULL)
            WITH e, rand() as random_value
            ORDER BY random_value
            LIMIT 1
            RETURN COALESCE(e.name, e.title) as entity_name,
                   e.description as description,
                   labels(e) as labels
            """
            
            result = session.run(query)
            record = result.single()
            
            if not record:
                await update.message.reply_text("Hmm, I couldn't find any entities to explore right now. Try again!")
                return
            
            entity_name = record["entity_name"]
            description = record.get("description", "")
            labels = record.get("labels", [])
            
            # Build a question about this entity
            question = f"Tell me about {entity_name}"
            
            logger.info(f"[RANDOM] User {user_id} got random entity: {entity_name}")
            
            # Load conversation history
            conversation_history = get_chat_history_from_graph(user_id, limit_pairs=10)
            recent_entities = get_recent_entities_from_graph(user_id, limit=20)
            
            # Get detailed response using the chatbot engine
            response, discussed_entities = chat_with_kg(question, GRAPH_SCHEMA, conversation_history, recent_entities)
            
            # Add a header to indicate this was random
            random_header = f"🎲 Random Discovery: {entity_name}\n\n"
            full_response = random_header + response
            
            # Save to chat history
            user_info = {
                'username': update.effective_user.username or '',
                'first_name': update.effective_user.first_name or '',
                'last_name': update.effective_user.last_name or ''
            }
            save_message_to_graph(user_id, "user", f"/random (discovered: {entity_name})", [entity_name], user_info)
            save_message_to_graph(user_id, "assistant", full_response, discussed_entities, user_info)
            
            # Send response
            await update.message.reply_text(full_response)
            logger.info(f"Successfully sent random entity info to user {user_id}")
            
    except Exception as e:
        logger.error(f"Error in random command: {e}")
        await update.message.reply_text("Oops! I had trouble finding a random topic. Try again!")


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /reset command - clear conversation history from Memgraph using delete_chat_history logic."""
    user_id = update.effective_user.id
    first_name = update.effective_user.first_name or 'there'
    
    try:
        with driver.session() as session:
            # Delete messages for this user (same logic as delete_chat_history.py)
            delete_query = """
            MATCH (u:User {user_id: $user_id})-[s:SENT]->(m:Message)
            DETACH DELETE m, s
            """
            session.run(delete_query, user_id=user_id)
            
            # Delete user node - simpler approach without pattern matching
            # Check if user still has any relationships first
            check_query = """
            MATCH (u:User {user_id: $user_id})
            OPTIONAL MATCH (u)-[r]-()
            WITH u, count(r) as rel_count
            WHERE rel_count = 0
            DELETE u
            """
            session.run(check_query, user_id=user_id)
            
            logger.info(f"Cleared chat history for user {user_id} via /reset command")
            await update.message.reply_text(f"✨ {first_name}, your chat history has been cleared! Let's start fresh. What would you like to know?")
            
    except Exception as e:
        logger.error(f"Error clearing chat history for user {user_id}: {e}")
        await update.message.reply_text("Sorry, I encountered an error while clearing your history. Please try again.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming messages."""
    try:
        # Validate update
        if not update.message or not update.message.text:
            logger.warning("Received update without message or text")
            return
        
        user_id = update.effective_user.id
        user_message = update.message.text
        
        # Extract user info from Telegram
        user_info = {
            'username': update.effective_user.username or '',
            'first_name': update.effective_user.first_name or '',
            'last_name': update.effective_user.last_name or ''
        }
        
        logger.info(f"User {user_id} ({user_info['first_name']}): {user_message}")
        
        # Send typing indicator
        await update.message.chat.send_action(action="typing")
        
        # Load conversation history from Memgraph (last 10 Q&A pairs)
        conversation_history = get_chat_history_from_graph(user_id, limit_pairs=10)
        recent_entities = get_recent_entities_from_graph(user_id, limit=20)
        
        logger.info(f"Loaded {len(conversation_history)} messages and {len(recent_entities)} entities from history")
        
        # Get response from chatbot with context
        response, discussed_entities = chat_with_kg(user_message, GRAPH_SCHEMA, conversation_history, recent_entities)
        
        # Log query-answer pair for training data
        log_training_data(user_id, user_message, response, user_info)
        
        # Save user message and bot response to Memgraph
        save_message_to_graph(user_id, "user", user_message, discussed_entities, user_info)
        save_message_to_graph(user_id, "assistant", response, discussed_entities, user_info)
        
        # Send response
        await update.message.reply_text(response)
        logger.info(f"Successfully responded to user {user_id}")
        
    except Exception as e:
        logger.error(f"Error handling message from user {user_id}: {e}", exc_info=True)
        try:
            await update.message.reply_text(
                "Oops! I hit a snag there. Could you try asking in a different way?"
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {send_error}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")


def main() -> None:
    """Start the Telegram bot."""
    global GRAPH_SCHEMA, FAISS_INDEX, ENTITY_LIST, ENTITY_METADATA
    
    print("=" * 80)
    print("🌿 Kuchiko - Telegram Knowledge Graph Chatbot (HYBRID RAG) 🌿")
    print("=" * 80)
    print("Initializing...")
    
    # Initialize chat history schema in Memgraph
    print("Setting up chat history persistence...")
    init_chat_schema()
    
    # Load schema once at startup
    print("Loading knowledge graph schema...")
    GRAPH_SCHEMA = get_graph_schema()
    logger.info("Schema loaded successfully")
    
    # Initialize FAISS index for hybrid search
    print("Initializing FAISS vector index...")
    FAISS_INDEX, ENTITY_LIST, ENTITY_METADATA = load_faiss_index()
    
    if FAISS_INDEX is None:
        print("No existing FAISS index found. Building new index...")
        print("This may take a few minutes...")
        entity_names, embeddings, metadata = build_entity_embeddings()
        
        if entity_names and embeddings:
            FAISS_INDEX = create_faiss_index(embeddings)
            if FAISS_INDEX:
                ENTITY_LIST = entity_names
                ENTITY_METADATA = metadata
                save_faiss_index(FAISS_INDEX, ENTITY_LIST, ENTITY_METADATA)
                print(f"✅ FAISS index built with {len(ENTITY_LIST)} entities")
        else:
            print("⚠️  Could not build FAISS index. Will use LLM-NLU fallback only.")
    else:
        print(f"✅ Loaded FAISS index with {len(ENTITY_LIST)} entities")
    
    print("Connecting to Telegram...")
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("table", table_command))
    application.add_handler(CommandHandler("random", random_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Register error handler
    application.add_error_handler(error_handler)
    
    print("=" * 80)
    print("✅ Bot is running! Press Ctrl+C to stop.")
    print()
    print("🔬 HYBRID RAG Architecture:")
    print(f"   • Vector Search: {len(ENTITY_LIST)} entities indexed" if FAISS_INDEX else "   • Vector Search: Disabled (using LLM-NLU only)")
    print(f"   • Graph Expansion: {MAX_HOPS} hops")
    print(f"   • Top-K Seeds: {TOP_K_SEEDS}")
    print(f"   • Max Subgraph Nodes: {MAX_SUBGRAPH_NODES}")
    print()
    print("💾 Chat history is persisted in Memgraph")
    print(f"🕐 Using Malaysia timezone (Asia/Kuala_Lumpur)")
    print(f"📝 Storing last 10 Q&A pairs per user")
    print("=" * 80)
    
    # Start the bot with drop_pending_updates=True to avoid processing old messages
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    
    # Cleanup
    driver.close()
    print("\nBot stopped. Connection closed.")


if __name__ == "__main__":
    main()
