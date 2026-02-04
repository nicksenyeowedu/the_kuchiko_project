# Kuchiko - Technical Architecture Guide

This document provides a detailed explanation of the entire pipeline from PDF ingestion to knowledge graph creation to the Hybrid RAG chatbot system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Stage 1: PDF Extraction & Chunking](#stage-1-pdf-extraction--chunking)
4. [Stage 2: Knowledge Graph Creation](#stage-2-knowledge-graph-creation)
5. [Stage 3: Embedding Generation & FAISS Index](#stage-3-embedding-generation--faiss-index)
6. [Stage 4: Hybrid RAG Retrieval](#stage-4-hybrid-rag-retrieval)
7. [Stage 5: Response Generation](#stage-5-response-generation)
8. [Data Flow Diagram](#data-flow-diagram)
9. [Technical Stack](#technical-stack)

---

## System Overview

Kuchiko is a **Hybrid RAG (Retrieval-Augmented Generation)** chatbot that combines:

1. **Vector Search** - FAISS index for semantic similarity matching
2. **Graph Database** - Memgraph for structured knowledge storage
3. **LLM** - NVIDIA NIM (DeepSeek) for entity extraction and response generation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           KUCHIKO ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐ │
│   │   PDF   │ ──► │  Chunking &  │ ──► │  Knowledge  │ ──► │   FAISS     │ │
│   │  Input  │     │  Extraction  │     │    Graph    │     │   Index     │ │
│   └─────────┘     └──────────────┘     └─────────────┘     └─────────────┘ │
│                                               │                     │       │
│                                               ▼                     ▼       │
│                                        ┌─────────────────────────────────┐  │
│                                        │      Hybrid RAG Retrieval       │  │
│   ┌─────────┐                          │  ┌───────────┐ ┌─────────────┐  │  │
│   │  User   │ ──────────────────────►  │  │  Vector   │ │    Graph    │  │  │
│   │  Query  │                          │  │  Search   │ │  Expansion  │  │  │
│   └─────────┘                          │  └───────────┘ └─────────────┘  │  │
│                                        └─────────────────────────────────┘  │
│                                                        │                     │
│                                                        ▼                     │
│   ┌─────────┐                          ┌─────────────────────────────────┐  │
│   │  Bot    │ ◄────────────────────────│      LLM Response Generation   │  │
│   │Response │                          │        (NVIDIA NIM)             │  │
│   └─────────┘                          └─────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Architecture

The system operates in two phases:

### Phase 1: Offline Processing (Build Time)
```
PDF → Chunking → Entity Extraction → Knowledge Graph → Embedding Index
```

### Phase 2: Online Query (Runtime)
```
User Query → Vector Search → Graph Expansion → Subgraph Pruning → LLM Synthesis → Response
```

---

## Stage 1: PDF Extraction & Chunking

**File:** `createKG.py`

### 1.1 PDF Text Extraction

```python
def extract_pdf_pages(path: str) -> List[str]:
    """Return a list of page texts (in order)."""
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        text = p.extract_text()
        pages.append(text.strip())
    return pages
```

**Process:**
1. Open PDF using `pypdf`
2. Extract text from each page sequentially
3. Preserve page order for later reference

### 1.2 Embedding-Based Page Clustering

Instead of fixed-size chunking, we use **semantic clustering** to group related pages:

```python
def recursive_cluster_pages(pages: List[str]) -> List[List[int]]:
    """Recursively cluster pages based on embedding similarity."""

    # Generate embeddings for page summaries (first 500 chars)
    page_embeddings = []
    for p in pages:
        summary = p[:500] if p else ""
        emb = get_embedding(summary)  # Using sentence-transformers
        page_embeddings.append(emb)

    # Group consecutive similar pages
    clusters = []
    current_cluster = [0]

    for i in range(1, len(pages)):
        sim = cosine_similarity(page_embeddings[i], page_embeddings[i-1])
        if sim > 0.7:  # Continuity threshold
            current_cluster.append(i)
        else:
            clusters.append(current_cluster)
            current_cluster = [i]

    return clusters
```

**Key Concepts:**
- **Embedding Model:** `all-MiniLM-L6-v2` (sentence-transformers)
- **Similarity Threshold:** 0.7 (pages above this are considered continuous)
- **Output:** Groups of page indices that belong together semantically

### 1.3 Semantic Boundary Refinement

Within each cluster, we ask the LLM to split text into coherent sections:

```python
def refine_boundaries(text: str) -> List[Dict[str, Any]]:
    """Ask NVIDIA NIM to split merged text into semantically coherent sections."""

    prompt = """
    You are a document analyzer. Split the input text into semantically coherent sections.
    Return a JSON array where each element is an object with keys: "title" and "content".
    """

    resp = ask_nim(prompt + text[:30000])
    return parse_json(resp)  # Returns [{title: "...", content: "..."}, ...]
```

**Chunking Strategy:**
```
┌────────────────────────────────────────────────────────────────────┐
│                         PDF Document                                │
├────────────────────────────────────────────────────────────────────┤
│  Page 1  │  Page 2  │  Page 3  │  Page 4  │  Page 5  │  Page 6    │
├──────────┴──────────┼──────────┴──────────┼──────────┴────────────┤
│      Segment 1      │      Segment 2      │      Segment 3        │
│  (Pages 1-2)        │  (Pages 3-4)        │  (Pages 5-6)          │
├─────────┬───────────┼─────────┬───────────┼─────────┬─────────────┤
│Section A│Section B  │Section C│Section D  │Section E│Section F    │
│(LLM-    │(LLM-      │(LLM-    │(LLM-      │(LLM-    │(LLM-        │
│refined) │refined)   │refined) │refined)   │refined) │refined)     │
└─────────┴───────────┴─────────┴───────────┴─────────┴─────────────┘
```

---

## Stage 2: Knowledge Graph Creation

**File:** `createKG.py`

### 2.1 Entity & Relationship Extraction

For each section, we use the LLM to extract structured knowledge:

```python
def extract_kg(text: str, section_title: str = None) -> Dict[str, Any]:
    """Ask NVIDIA NIM to extract KG with dynamic ontology discovery."""

    prompt = """Extract a knowledge graph from the text. Be PRECISE and LITERAL.

    STRICT RULES:
    1. Extract entities: people, places, organizations, ethnic groups, events, dates, concepts
    2. Extract relationships ONLY when the text directly states a connection
    3. Use meaningful relationship types: CONTROLLED_BY, RULED_BY, LOCATED_IN, etc.
    4. Evidence must be the EXACT sentence proving the relationship

    Return JSON:
    {
      "entities": [
        {"id": "e1", "type": "TYPE", "name": "entity name",
         "attributes": {}, "mentions": ["exact sentence..."]}
      ],
      "relationships": [
        {"source": "e1", "relation": "RELATION_TYPE", "target": "e2",
         "evidence": ["exact sentence..."]}
      ]
    }"""

    return parse_json(ask_nim(prompt + text))
```

**Example Extraction:**
```
Input Text: "the Bau bazaar controlled by Hakka miners"

Output:
{
  "entities": [
    {"id": "e1", "type": "LOCATION", "name": "Bau bazaar"},
    {"id": "e2", "type": "ETHNIC_GROUP", "name": "Hakka miners"}
  ],
  "relationships": [
    {
      "source": "e1",
      "relation": "CONTROLLED_BY",
      "target": "e2",
      "evidence": ["the Bau bazaar controlled by Hakka miners"]
    }
  ]
}
```

### 2.2 Entity Deduplication

Entities with similar names are merged using embedding similarity:

```python
def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """Merge similar entities based on embedding similarity with semantic checks."""

    SIMILARITY_THRESHOLD = 0.95  # Very high - only merge near-identical names

    for each entity pair:
        # RULE 1: Only merge if types match
        if primary_type != other_type:
            continue

        # RULE 2: Check substring relationship
        has_substring = other_lower in primary_lower or primary_lower in other_lower

        # RULE 3: Check for conflicting keywords
        direction_keywords = ['downstream', 'upstream', 'north', 'south', ...]
        has_conflicting_direction = check_conflicts(other_lower, primary_lower)

        # Only merge if similarity > 0.95 AND passes semantic checks
        if should_merge:
            merge(other -> primary)
```

**Deduplication Rules:**
| Rule | Description | Example |
|------|-------------|---------|
| Type Match | Only merge entities of same type | Person + Person ✓, Person + Location ✗ |
| High Similarity | Cosine similarity > 0.95 | "James Brooke" ≈ "Brooke, James" |
| Substring Check | One name contains the other | "White Rajah" ⊂ "White Rajah kingdom" |
| Conflict Detection | Don't merge opposing terms | "downstream" ≠ "upstream" |

### 2.3 Graph Storage in Memgraph

The hierarchical structure stored in Memgraph:

```
Document (root)
    │
    ├── Segment (pages 1-3)
    │       │
    │       ├── Entity: "James Brooke" (Person)
    │       │       └── Mention: "James Brooke was the first..."
    │       │
    │       ├── Entity: "Sarawak" (Location)
    │       │
    │       └── Relationship: James Brooke -[RULED]-> Sarawak
    │
    └── Segment (pages 4-6)
            │
            └── ...
```

**Cypher Insert Example:**
```cypher
// Create Document node
MERGE (doc:Document {id:$doc_id})

// Create Segment node linked to Document
MERGE (seg:Segment {segment_id:$segment_id})
MERGE (doc)-[:CONTAINS_SEGMENT]->(seg)

// Create Entity node with embedding
MERGE (n:Entity {name:$name, type:$type})
SET n.embedding=$embedding
MERGE (seg)-[:CONTAINS_ENTITY]->(n)

// Create Relationship between entities
MATCH (a:Entity {id:$src}) MATCH (b:Entity {id:$tgt})
MERGE (a)-[x:RULED]->(b)
SET x.evidence = $evidence
```

---

## Stage 3: Embedding Generation & FAISS Index

**File:** `build_embeddings.py`

### 3.1 Entity Embedding

After the knowledge graph is built, we generate embeddings for all entities:

```python
def extract_entities_from_kg():
    """Extract all entities from knowledge graph."""

    query = """
    MATCH (n)
    WHERE NOT n:User AND NOT n:Message
    AND (n.name IS NOT NULL)
    RETURN
        COALESCE(n.name, n.id, n.title) as entity_name,
        labels(n) as labels,
        properties(n) as props
    """

    # Build rich text representation for embedding
    # Format: "Entity Name | Description | Type: Label"
    text_parts = [entity_name]
    if props.get("description"):
        text_parts.append(description[:500])
    if labels:
        text_parts.append(f"Type: {', '.join(labels)}")

    embedding_text = " | ".join(text_parts)
    return embedding_text
```

### 3.2 NVIDIA NIM Embedding Model

```python
NIM_EMBEDDING_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts in a single API call."""

    response = nim_client.embeddings.create(
        input=texts,
        model=NIM_EMBEDDING_MODEL,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"}
    )
    return [item.embedding for item in response.data]
```

### 3.3 FAISS Index Construction

```python
def build_faiss_index(embeddings: List[List[float]]) -> faiss.Index:
    """Build FAISS index from embeddings."""

    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]

    # Use L2 distance (Euclidean)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    return index
```

**Output Files:**
| File | Content |
|------|---------|
| `faiss_index.bin` | FAISS vector index |
| `entities.pkl` | List of entity names (maps index position → name) |
| `entity_metadata.pkl` | Entity metadata (name → properties) |

---

## Stage 4: Hybrid RAG Retrieval

**File:** `chatbot_telegram.py`

### 4.1 Query Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HYBRID RAG RETRIEVAL                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Query: "Who was James Brooke?"                                       │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Step 1: INTENT CLASSIFICATION                                        │   │
│   │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│   │ │ Categories: GREETING | SMALL_TALK | FOLLOW_UP | KNOWLEDGE_QUERY │ │   │
│   │ │             META                                                 │ │   │
│   │ │ Result: KNOWLEDGE_QUERY (needs_kg=True)                         │ │   │
│   │ └─────────────────────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Step 2: VECTOR SEARCH (FAISS)                                        │   │
│   │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│   │ │ 1. Embed query using NIM embedding model                        │ │   │
│   │ │ 2. Search FAISS index for top-K similar entities                │ │   │
│   │ │ 3. Return seed entities: ["James Brooke", "White Rajah", ...]   │ │   │
│   │ └─────────────────────────────────────────────────────────────────┘ │   │
│   │ Parameters: TOP_K_SEEDS = 5                                         │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Step 3: GRAPH EXPANSION                                              │   │
│   │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│   │ │ Starting from seed entities, expand N hops in the graph         │ │   │
│   │ │                                                                  │ │   │
│   │ │     [James Brooke] ──RULED──► [Sarawak]                         │ │   │
│   │ │           │                       │                              │ │   │
│   │ │           ▼                       ▼                              │ │   │
│   │ │     [White Rajah]          [Kuching]                            │ │   │
│   │ │           │                       │                              │ │   │
│   │ │           ▼                       ▼                              │ │   │
│   │ │     [Brooke Dynasty]      [Sarawak River]                       │ │   │
│   │ └─────────────────────────────────────────────────────────────────┘ │   │
│   │ Parameters: MAX_HOPS = 2                                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                     │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Step 4: SUBGRAPH RANKING & PRUNING                                   │   │
│   │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│   │ │ Score each node based on:                                       │ │   │
│   │ │   • Is seed entity? (+100 points)                               │ │   │
│   │ │   • Distance from seeds (closer = higher score)                 │ │   │
│   │ │   • Has description/text? (+20 points)                          │ │   │
│   │ │   • Degree centrality (more connections = higher score)         │ │   │
│   │ │                                                                  │ │   │
│   │ │ Keep top-N nodes by score                                       │ │   │
│   │ └─────────────────────────────────────────────────────────────────┘ │   │
│   │ Parameters: MAX_SUBGRAPH_NODES = 50                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                     │
│        ▼                                                                     │
│   [Pruned Subgraph with relevant context for LLM]                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Vector Search Implementation

```python
def vector_search_seeds(query: str, top_k: int = 5) -> List[str]:
    """Search for top-K seed entities using vector similarity."""

    # 1. Embed the query
    query_embedding = get_embedding(query, input_type="query")
    query_vector = np.array([query_embedding]).astype('float32')

    # 2. Search FAISS index
    distances, indices = FAISS_INDEX.search(query_vector, top_k)

    # 3. Map indices to entity names
    seed_entities = [ENTITY_LIST[idx] for idx in indices[0]]

    return seed_entities
```

### 4.3 Graph Expansion

```python
def expand_from_seeds(seed_entities: List[str], max_hops: int = 2):
    """Expand graph from seed entities using multi-hop traversal."""

    query = f"""
    MATCH path = (seed)-[*1..{max_hops}]-(connected)
    WHERE seed.name IN $seeds
    RETURN seed, connected, relationships(path) as rels
    LIMIT 200
    """

    # Build subgraph with nodes and relationships
    nodes = {}  # node_id -> {properties, distance, is_seed}
    relationships = []  # [{source, target, types, distance}]

    return {"nodes": nodes, "relationships": relationships}
```

### 4.4 Fallback: LLM-NLU

If vector search fails, we fall back to LLM-based keyword extraction:

```python
def extract_keywords(question: str, schema: str) -> Dict[str, Any]:
    """Use LLM as NLU to extract keywords and query intent."""

    prompt = """Analyze the user's question and extract keywords.
    Return JSON:
    {
        "keywords": ["keyword1", "keyword2", ...],
        "entity_types": ["Person", "Location", ...],
        "query_intent": "search|relationship|property|list",
        "confidence": 0.0-1.0
    }

    IMPORTANT: Include both singular and plural forms of keywords.
    """

    return parse_json(ask_nim(prompt + question))
```

---

## Stage 5: Response Generation

### 5.1 Context Formatting

The pruned subgraph is formatted as text for the LLM:

```python
# Format subgraph as triples for LLM
context_text = f"Seed Entities: {', '.join(seeds)}\n\n"
context_text += "Knowledge Graph Subgraph:\n"

# Add node information
for node_id, node_data in nodes.items():
    desc = node_data.get("description", "")
    context_text += f"- {node_id}: {desc[:200]}\n"

# Add relationships as triples
context_text += "\nRelationships:\n"
for rel in relationships:
    context_text += f"- {rel['source']} → {rel['types']} → {rel['target']}\n"
```

### 5.2 LLM Synthesis

```python
system_prompt = """You are Kuchiko, a knowledgeable AI assistant.

⚠️ CRITICAL: You MUST ONLY use information from the provided context data.
DO NOT use your own knowledge or make assumptions.

You have access to structured data from a knowledge graph:
1. PRIMARY data: Direct factual information about the user's query
2. SECONDARY data: 2-hop neighbors for exploration suggestions

Response structure:
1. Provide a concise, conversational answer using PRIMARY data
2. Add exploration suggestions from SECONDARY data
"""

response = nim_client.chat.completions.create(
    model="deepseek-ai/deepseek-v3.1",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ],
    temperature=0.8,
    max_tokens=1000
)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ═══════════════════════ OFFLINE PROCESSING ═══════════════════════        │
│                                                                              │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  PDF    │───►│   pypdf     │───►│  Embedding  │───►│   Page      │      │
│  │  File   │    │  Extract    │    │  Clustering │    │  Segments   │      │
│  └─────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                              │               │
│                                                              ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LLM Entity Extraction                             │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │  Section    │───►│  NVIDIA NIM │───►│  Entities + Relations   │  │   │
│  │  │  Text       │    │  (DeepSeek) │    │  JSON                   │  │   │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                              │               │
│                                                              ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Entity Deduplication                              │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │  Entities   │───►│  sentence-  │───►│  Merged Unique          │  │   │
│  │  │  (with      │    │  transformers│   │  Entities               │  │   │
│  │  │  duplicates)│    │  Similarity │    │                         │  │   │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                              │               │
│                              ┌────────────────────────────────┤               │
│                              │                                ▼               │
│                              │    ┌─────────────────────────────────────┐   │
│                              │    │            Memgraph                  │   │
│                              │    │  ┌─────────────────────────────────┐│   │
│                              │    │  │ (Document)──[:CONTAINS]──►      ││   │
│                              │    │  │   (Segment)──[:CONTAINS]──►     ││   │
│                              │    │  │     (Entity)──[:RELATION]──►    ││   │
│                              │    │  │       (Entity)                  ││   │
│                              │    │  └─────────────────────────────────┘│   │
│                              │    └─────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FAISS Index Building                              │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │  Entity     │───►│  NVIDIA NIM │───►│  FAISS Index            │  │   │
│  │  │  Names +    │    │  Embedding  │    │  + Entity Mapping       │  │   │
│  │  │  Metadata   │    │  API        │    │                         │  │   │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ════════════════════════ ONLINE QUERY ════════════════════════════        │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐     │
│  │  Telegram   │───►│  Intent     │───►│  KNOWLEDGE_QUERY detected   │     │
│  │  User Query │    │  Classifier │    │                             │     │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘     │
│                                                        │                     │
│                                                        ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Hybrid RAG Retrieval                              │   │
│  │                                                                       │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │  Query      │───►│  FAISS      │───►│  Seed Entities          │  │   │
│  │  │  Embedding  │    │  Search     │    │  (Top-K)                │  │   │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  │                                                        │              │   │
│  │                                                        ▼              │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │  Seed       │───►│  Memgraph   │───►│  Expanded Subgraph      │  │   │
│  │  │  Entities   │    │  Multi-hop  │    │  (Nodes + Relations)    │  │   │
│  │  └─────────────┘    │  Traversal  │    └─────────────────────────┘  │   │
│  │                     └─────────────┘               │                  │   │
│  │                                                   ▼                  │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐│   │
│  │  │  Ranking & Pruning                                              ││   │
│  │  │  • Score by: seed status, distance, description, degree         ││   │
│  │  │  • Keep top-N nodes                                             ││   │
│  │  └─────────────────────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                        │                     │
│                                                        ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LLM Response Generation                           │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │  │  Pruned     │───►│  NVIDIA NIM │───►│  Natural Language       │  │   │
│  │  │  Subgraph   │    │  (DeepSeek) │    │  Response               │  │   │
│  │  │  Context    │    │             │    │                         │  │   │
│  │  └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                        │                     │
│                                                        ▼                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Telegram Bot Response                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **PDF Processing** | pypdf | Extract text from PDF documents |
| **Local Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Page clustering, entity deduplication |
| **Cloud Embeddings** | NVIDIA NIM (llama-3.2-nemoretriever-300m) | Entity embeddings for FAISS |
| **LLM** | NVIDIA NIM (deepseek-ai/deepseek-v3.1) | Entity extraction, response generation |
| **Vector Index** | FAISS (IndexFlatL2) | Fast similarity search |
| **Graph Database** | Memgraph | Knowledge graph storage |
| **Chat Interface** | python-telegram-bot | Telegram bot API |
| **Orchestration** | Docker Compose | Container management |

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | 0.95 | Minimum similarity for entity deduplication |
| `TOP_K_SEEDS` | 5 | Number of seed entities from vector search |
| `MAX_HOPS` | 2 | Maximum hops for graph expansion |
| `MAX_SUBGRAPH_NODES` | 50 | Maximum nodes in pruned subgraph |
| `BATCH_SIZE` | 50 | Entities per embedding API call |

---

## Summary

The Kuchiko pipeline combines multiple AI techniques:

1. **Semantic Chunking** - Uses embeddings to group related pages
2. **LLM Extraction** - Structured entity/relationship extraction
3. **Graph Storage** - Hierarchical knowledge representation
4. **Hybrid Retrieval** - Vector search + graph traversal
5. **Contextual Generation** - LLM synthesis with grounding

This architecture ensures:
- **Accuracy**: Information is grounded in the knowledge graph
- **Relevance**: Vector search finds semantically similar entities
- **Context**: Graph expansion provides related information
- **Natural responses**: LLM generates conversational answers
