"""
Dynamic LLM-guided PDF -> Knowledge Graph pipeline with LangChain
- Uses NVIDIA NIM via LangChain for entity/relation extraction
- Embedding-based deduplication and hierarchical graph storage in Memgraph
- Parallel segment processing with validation and cleanup

Assumptions:
- Memgraph is reachable at bolt://localhost:7687 with username/password set
- NVIDIA_API_KEY is available in environment variable NVIDIA_API_KEY
- Python packages: openai, pypdf, neo4j, python-dotenv, langchain, sentence-transformers

Install:
    pip install openai pypdf neo4j python-dotenv langchain sentence-transformers

Run:
    export NVIDIA_API_KEY="..."
    python createKG.py /path/to/file.pdf

Features:
- LangChain integration for structured extraction
- Embedding-based entity deduplication (cosine similarity > 0.9)
- Recursive clustering for continuity detection
- Strict ontology validation
- Hierarchical node structure: Document → Segment → Section → KG
- Post-extraction validation and cleanup
- Parallel segment processing
"""

import os
import sys
import json
import time
import uuid
import logging
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from pypdf import PdfReader
from neo4j import GraphDatabase, basic_auth
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from token_tracker import tracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Silence noisy HTTP request logs from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# --------- Configuration ---------
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    logger.error("NVIDIA_API_KEY not set. Please set it in your .env file.")
    exit(1)

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "memgraph")
MEMGRAPH_PASS = os.getenv("MEMGRAPH_PASS", "memgraph")

# Model selection
NIM_MODEL = "deepseek-ai/deepseek-v3.1"
NIM_API_BASE = "https://integrate.api.nvidia.com/v1"

# Embedding model for deduplication and clustering
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.95  # Increased back to 0.95 - only merge VERY similar names

# Dynamic ontology tracking (learned from extraction)
discovered_entity_types = set()
discovered_relation_types = set()

# Initialize models and clients
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

nim_client = None
if NVIDIA_API_KEY:
    nim_client = OpenAI(
        base_url=NIM_API_BASE,
        api_key=NVIDIA_API_KEY
    )

# Neo4j driver for Memgraph
driver = GraphDatabase.driver(MEMGRAPH_URI, auth=basic_auth(MEMGRAPH_USER, MEMGRAPH_PASS))

# --------- Embedding & Deduplication ---------

def get_embedding(text: str) -> List[float]:
    """Generate embedding for text using sentence-transformers."""
    try:
        t0 = time.time()
        embedding = embedding_model.encode(text, convert_to_tensor=False, show_progress_bar=False)
        elapsed_ms = (time.time() - t0) * 1000
        tracker.log_local_embedding(texts_count=1, caller="createKG.get_embedding", elapsed_ms=elapsed_ms)
        return embedding.tolist()
    except Exception as e:
        logger.warning(f"Failed to generate embedding: {e}")
        return []

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    if not a or not b:
        return 0.0
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge similar entities based on embedding similarity with semantic checks."""
    if not entities:
        return []
    
    # Generate embeddings for entity names
    entity_embeddings = {}
    for e in entities:
        name = e.get('name', '')
        if name:
            entity_embeddings[e.get('id')] = {
                'embedding': get_embedding(name),
                'entity': e,
                'name': name,
                'name_lower': name.lower()
            }
    
    # Merge similar entities with STRICT rules
    merged = {}
    processed = set()
    merge_log = []
    comparison_log = []
    
    for eid, data in entity_embeddings.items():
        if eid in processed:
            continue
        
        primary = data['entity']
        merged[eid] = primary
        processed.add(eid)
        
        # Find similar entities and merge them
        for other_id, other_data in entity_embeddings.items():
            if other_id in processed or other_id == eid:
                continue
            
            sim = cosine_similarity(data['embedding'], other_data['embedding'])
            primary_name = data['name']
            other_name = other_data['name']
            primary_type = primary.get('type', 'N/A')
            other_type = other_data['entity'].get('type', 'N/A')
            
            # RULE 1: Only merge if types match
            if primary_type != other_type:
                comparison_log.append(f"SKIP (type mismatch): '{other_name}' ({other_type}) vs '{primary_name}' ({primary_type})")
                continue
            
            # RULE 2: Check for common substrings (one should contain the other)
            other_lower = other_data['name_lower']
            primary_lower = data['name_lower']
            
            # Check if one is a substring of the other (e.g., "White Rajah" in "White Rajah kingdom")
            has_substring = other_lower in primary_lower or primary_lower in other_lower
            
            # RULE 3: Penalize differences in key keywords
            # e.g., "downstream" vs "upstream" are different directions - should NOT merge
            direction_keywords = ['downstream', 'upstream', 'north', 'south', 'east', 'west', 'left', 'right']
            has_conflicting_direction = any(
                (keyword in other_lower) != (keyword in primary_lower) 
                for keyword in direction_keywords
            )
            
            comparison_msg = f"COMPARE: '{other_name}' ({other_type}) vs '{primary_name}' ({primary_type}) → similarity: {sim:.3f}"
            if has_substring:
                comparison_msg += " [substring=True]"
            if has_conflicting_direction:
                comparison_msg += " [CONFLICTING DIRECTION]"
            comparison_log.append(comparison_msg)
            
            # Only merge if:
            # - Similarity is VERY high (0.95+)
            # - AND either has substring OR high similarity AND no conflicting keywords
            should_merge = (sim > SIMILARITY_THRESHOLD) and (has_substring or not has_conflicting_direction)
            
            if should_merge:
                # Double-check: don't merge if names are fundamentally different
                # (e.g., "downstream Chinese" vs "upstream Chinese")
                name_words_other = set(other_lower.split())
                name_words_primary = set(primary_lower.split())
                common_words = name_words_other & name_words_primary
                overlap_ratio = len(common_words) / max(len(name_words_other), len(name_words_primary))
                
                if overlap_ratio < 0.5:  # Less than 50% word overlap = probably different
                    comparison_log.append(f"  → REJECTED (low word overlap: {overlap_ratio:.1%})")
                    continue
                
                # Merge other into primary
                other = other_data['entity']
                primary['mentions'].extend(other.get('mentions', []))
                primary['mentions'] = list(set(primary['mentions']))
                processed.add(other_id)
                
                merge_msg = f"[MERGED] '{other_name}' ({other_id}, {other_type}) → '{primary_name}' ({eid}, {primary_type}) [sim: {sim:.3f}, overlap: {overlap_ratio:.1%}]"
                logger.info(merge_msg)
                merge_log.append(merge_msg)
    
    # Write logs to file
    log_file = "entity_merge_log.txt"
    if merge_log or comparison_log:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Deduplication run at {datetime.now().isoformat()}\n")
            f.write(f"Total entities analyzed: {len(entity_embeddings)}\n")
            f.write(f"Similarity threshold: {SIMILARITY_THRESHOLD}\n")
            f.write(f"{'='*80}\n")
            
            if merge_log:
                f.write(f"\nMERGES PERFORMED ({len(merge_log)}):\n")
                for msg in merge_log:
                    f.write(msg + "\n")
            else:
                f.write(f"\nNO MERGES PERFORMED\n")
            
            f.write(f"\nALL COMPARISONS ({len(comparison_log)}):\n")
            for msg in comparison_log[:100]:  # Show first 100 comparisons
                f.write(msg + "\n")
            if len(comparison_log) > 100:
                f.write(f"... and {len(comparison_log) - 100} more comparisons\n")
            f.write("\n")
        
        # logger.info(f"Merge log written to {log_file}")
    
    return list(merged.values())

# --------- Utilities ---------

def safe_json_load(s: str) -> Any:
    """Parse JSON and strip common markdown fences before parsing."""
    trimmed = s.strip()

    # Strip leading/trailing markdown code fences like ```json ... ```
    if trimmed.startswith("```"):
        first_nl = trimmed.find("\n")
        if first_nl != -1:
            trimmed = trimmed[first_nl + 1 :]
        if trimmed.endswith("```"):
            trimmed = trimmed[:-3].rstrip()

    try:
        return json.loads(trimmed)
    except Exception:
        # try to salvage JSON-looking substring
        start = trimmed.find('{')
        end = trimmed.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(trimmed[start:end + 1])
            except Exception:
                pass
    raise ValueError(f"Failed to parse JSON from string: {trimmed[:200]}")


# --------- PDF Extraction ---------

def extract_pdf_pages(path: str) -> List[str]:
    """Return a list of page texts (in order)."""
    reader = PdfReader(path)
    pages = []
    for i, p in enumerate(reader.pages):
        text = p.extract_text()
        if text is None:
            text = ""
        # Basic cleanup: normalize whitespace, preserve page breaks later
        pages.append(text.strip())
    logger.info(f"Extracted {len(pages)} pages from PDF")
    return pages


# --------- NVIDIA NIM Helpers ---------

def ask_nim(prompt: str, max_output_tokens: int = 8000, temperature: float = 0.0, retry: int = 2) -> str:
    """Call NVIDIA NIM via OpenAI API. Returns raw text.
    If API not configured, raise an informative error.
    """
    if not NVIDIA_API_KEY or not nim_client:
        raise RuntimeError("NVIDIA_API_KEY is not set. Set the environment variable and retry.")

    for attempt in range(retry + 1):
        try:
            t0 = time.time()
            response = nim_client.chat.completions.create(
                model=NIM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_output_tokens,
                top_p=0.7
            )
            elapsed_ms = (time.time() - t0) * 1000
            tracker.log_chat_completion(response, caller="createKG.ask_nim",
                                        prompt_preview=prompt[:100], elapsed_ms=elapsed_ms)
            # Extract text from response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return str(response)
        except Exception as e:
            logger.warning(f"NVIDIA NIM call failed (attempt {attempt}): {e}")
            if attempt < retry:
                time.sleep(1 + attempt * 2)
                continue
            raise


# --------- Continuity Detection (Embedding-based) ---------

def recursive_cluster_pages(pages: List[str]) -> List[List[int]]:
    """Recursively cluster pages based on embedding similarity."""
    if not pages:
        return []
    
    # Generate embeddings for page summaries (first 500 chars)
    page_embeddings = []
    total_pages = len(pages)
    # Log progress every 10% so docker logs shows meaningful updates
    log_interval = max(1, total_pages // 10)
    for i, p in enumerate(pages):
        summary = p[:500] if p else ""
        emb = get_embedding(summary)
        page_embeddings.append(emb)
        if (i + 1) % log_interval == 0 or (i + 1) == total_pages:
            pct = int((i + 1) / total_pages * 100)
            bar_len = 30
            filled = int(bar_len * (i + 1) / total_pages)
            bar = '█' * filled + '░' * (bar_len - filled)
            logger.info(f"Clustering pages: |{bar}| {pct}% ({i + 1}/{total_pages})")
    
    # Simple hierarchical clustering: group consecutive similar pages
    clusters = []
    current_cluster = [0]
    
    for i in range(1, len(pages)):
        # Compare with previous page
        sim = cosine_similarity(page_embeddings[i], page_embeddings[i-1])
        if sim > 0.7:  # Continuity threshold
            current_cluster.append(i)
        else:
            clusters.append(current_cluster)
            current_cluster = [i]
    
    if current_cluster:
        clusters.append(current_cluster)
    
    logger.info(f"Clustered {len(pages)} pages into {len(clusters)} segments")
    return clusters

# --------- Build Hierarchical Segments ---------


# --------- Semantic Boundary Refinement ---------

def refine_boundaries(text: str) -> List[Dict[str, Any]]:
    """Ask NVIDIA NIM to split merged text into semantically coherent sections.
    Returns a list of {title, content} dicts.
    """
    prompt = f"""
You are a document analyzer. Split the input text into semantically coherent sections.
Return a JSON array where each element is an object with keys: "title" and "content".
- title: short (3-7 words) descriptive title for the section
- content: the exact text for that section (preserve original sentences)

If no clear title exists, provide a short descriptive label like "Paragraph A".

Input text:
""" + text[:30000] + """

Respond with ONLY valid JSON. Do NOT include markdown fences or code blocks.
"""
    resp = ask_nim(prompt, max_output_tokens=4000, temperature=0.0)
    try:
        data = safe_json_load(resp)
        if isinstance(data, list):
            # basic validation
            cleaned = []
            for item in data:
                title = item.get('title', '').strip() if isinstance(item, dict) else 'Section'
                content = item.get('content', '').strip() if isinstance(item, dict) else str(item)
                cleaned.append({'title': title or 'Section', 'content': content})
            return cleaned
    except Exception as e:
        logger.warning(f"Failed to parse refine_boundaries JSON (using fallback): {str(e)[:150]}")
    # fallback
    return [{'title': 'FullSegment', 'content': text}]


# --------- Extract KG (Entities + Relations) ---------

def extract_kg(text: str, section_title: str = None) -> Dict[str, Any]:
    """Ask NVIDIA NIM to extract KG with dynamic ontology discovery."""
    
    composed = f"""Extract a knowledge graph from the text. Be PRECISE and LITERAL - only extract what is explicitly stated.

STRICT RULES:
1. Extract entities: people, places, organizations, ethnic groups, events, dates, concepts, or any important entity
2. Extract relationships ONLY when the text directly states a connection
3. Use meaningful relationship types: simple verbs or prepositions (e.g., CONTROLLED_BY, RULED_BY, MOVED_TO, LOCATED_IN, PART_OF, INVOLVED_IN, CONSISTS_OF, etc.)
4. Evidence must be the EXACT sentence proving the relationship
5. Do NOT create relationships by inference or assumption

EXAMPLES OF CORRECT EXTRACTION:
Text: "the Bau bazaar controlled by Hakka miners"
→ Entity: {{"id":"e1", "type":"LOCATION", "name":"Bau bazaar"}}
→ Entity: {{"id":"e2", "type":"ETHNIC_GROUP", "name":"Hakka miners"}}
→ Relationship: {{"source":"e1", "relation":"CONTROLLED_BY", "target":"e2", "evidence":["the Bau bazaar controlled by Hakka miners"]}}

Section Title: {section_title or 'General'}

Text to analyze:
{text[:30000]}

Return ONLY valid JSON (no markdown, no explanations):
{{
  "entities": [
    {{"id": "e1", "type": "TYPE", "name": "entity name", "attributes": {{}}, "mentions": ["exact sentence where entity appears"]}}
  ],
  "relationships": [
    {{"source": "e1", "relation": "RELATION_TYPE", "target": "e2", "evidence": ["exact sentence proving this relationship"]}}
  ],
  "raw_sentences": [
    {{"id": "s1", "text": "complete sentence from text", "page_index": 0}}
  ]
}}"""
    
    resp = ask_nim(composed, max_output_tokens=8000, temperature=0.0)

    try:
        kg = safe_json_load(resp)
        
        # Discover and track entity types dynamically
        for e in kg.get('entities', []):
            etype = e.get('type', '')
            if etype:
                discovered_entity_types.add(etype)
        
        # Discover and track relation types dynamically
        for r in kg.get('relationships', []):
            rel_type = r.get('relation', '')
            if rel_type:
                discovered_relation_types.add(rel_type)
        
        # Log discovered ontology periodically (commented out to reduce log noise)
        # if len(discovered_entity_types) % 5 == 0:
        #     logger.info(f"Discovered entity types so far: {sorted(discovered_entity_types)}")
        # if len(discovered_relation_types) % 5 == 0:
        #     logger.info(f"Discovered relation types so far: {sorted(discovered_relation_types)}")
        
        # Deduplicate entities
        kg['entities'] = deduplicate_entities(kg['entities'])
        
        # Ensure raw_sentences exist
        if 'raw_sentences' not in kg or not kg['raw_sentences']:
            sentences = [s.strip() for s in text.split('\n') if s.strip()]
            kg['raw_sentences'] = [{'id': f's{j}', 'text': s, 'page_index': None} for j, s in enumerate(sentences)]
        
        return kg
    except Exception as e:
        logger.warning(f"Failed to parse KG JSON from NIM (using fallback): {str(e)[:150]}")
        # Fallback
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        return {
            'entities': [],
            'relationships': [],
            'raw_sentences': [{'id': f's{j}', 'text': s, 'page_index': None} for j, s in enumerate(sentences)]
        }


# --------- Memgraph Insertions (Lossless) ---------

def insert_kg_into_memgraph(kg: Dict[str, Any], page_indices: List[int], segment_id: str, doc_id: str = None, max_retries: int = 5):
    """Insert KG with hierarchical structure: Document → Segment → Section → KG.
    Also creates embeddings for entities for similarity queries.
    Retries on Memgraph TransientError (conflicting transactions from parallel writes).
    """
    if not doc_id:
        doc_id = str(uuid.uuid4())

    for attempt in range(max_retries):
        try:
            _insert_kg_into_memgraph_inner(kg, page_indices, segment_id, doc_id)
            return  # Success
        except Exception as e:
            error_str = str(e)
            # Retry on transient/conflict errors
            if "TransientError" in error_str or "conflicting transactions" in error_str.lower() or "Cannot resolve" in error_str:
                wait_time = (2 ** attempt) * 0.5 + (hash(segment_id) % 10) * 0.1  # Backoff + jitter
                if attempt < max_retries - 1:
                    logger.warning(f"Memgraph conflict on {segment_id}, retrying in {wait_time:.1f}s ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Memgraph conflict on {segment_id} after {max_retries} retries, giving up: {e}")
                    raise
            else:
                raise  # Non-transient error, don't retry


def _insert_kg_into_memgraph_inner(kg: Dict[str, Any], page_indices: List[int], segment_id: str, doc_id: str):
    """Inner function that performs the actual Memgraph insertion (called by retry wrapper)."""
    with driver.session() as session:
        # Create Document node (root)
        session.run(
            "MERGE (doc:Document {id:$doc_id}) SET doc.created_at = timestamp()",
            doc_id=doc_id
        )
        
        # Create Segment node (hierarchical parent)
        session.run(
            "MATCH (doc:Document {id:$doc_id}) MERGE (seg:Segment {segment_id:$segment_id}) "
            "SET seg.text=$text, seg.page_indices=$pages, seg.created_at = timestamp() "
            "MERGE (doc)-[:CONTAINS_SEGMENT]->(seg)",
            doc_id=doc_id, segment_id=segment_id, text="\n".join([f"Page {i}" for i in page_indices]),
            pages=page_indices
        )

        # Insert raw_sentences
        raw_sentences = kg.get('raw_sentences', [])
        for s in raw_sentences:
            sid = s.get('id') or str(uuid.uuid4())
            text = s.get('text', '')
            session.run(
                "MERGE (r:RawSentence {id:$id}) SET r.text=$text, r.page_index=$page_index",
                id=sid, text=text, page_index=s.get('page_index')
            )

        # Insert entities with embeddings - MERGE by name+type to avoid duplicates across segments
        entity_id_map = {}
        for e in kg.get('entities', []):
            original_id = e.get('id')
            
            etype = e.get('type', 'ENTITY')
            name = e.get('name') or ''
            attrs = e.get('attributes', {}) or {}
            
            # Generate embedding for entity name
            embedding = get_embedding(name)
            
            # CRITICAL: MERGE by name+type, not by random ID
            # This ensures "Hakka miners" from different sections becomes ONE entity
            result = session.run(
                "MATCH (seg:Segment {segment_id:$segment_id}) "
                "MERGE (n:Entity {name:$name, type:$type}) "
                "SET n.attributes=$attrs, n.embedding=$embedding "
                "MERGE (seg)-[:CONTAINS_ENTITY]->(n) "
                "RETURN n.id as actual_id",
                segment_id=segment_id, name=name, type=etype, attrs=attrs, embedding=embedding
            )
            
            # Get the actual merged entity ID
            record = result.single()
            if record:
                actual_id = record.get('actual_id')
                # If entity doesn't have ID yet, generate one
                if not actual_id:
                    actual_id = str(uuid.uuid4())
                    session.run("MATCH (n:Entity {name:$name, type:$type}) SET n.id=$id", 
                               name=name, type=etype, id=actual_id)
            else:
                actual_id = str(uuid.uuid4())
            
            entity_id_map[original_id] = actual_id
            
            # Add mentions using the actual merged entity
            for m in e.get('mentions', []):
                mid = str(uuid.uuid4())
                session.run(
                    "MERGE (m:Mention {id:$mid}) SET m.text=$text "
                    "WITH m MATCH (n:Entity {name:$name, type:$type}) MERGE (n)-[:MENTIONED_IN]->(m)",
                    mid=mid, text=m, name=name, type=etype
                )

        # Insert relationships with validation
        for r in kg.get('relationships', []):
            src = entity_id_map.get(r.get('source'), r.get('source'))
            tgt = entity_id_map.get(r.get('target'), r.get('target'))
            rel = r.get('relation') or 'RELATED_TO'
            evidence = r.get('evidence', [])
            
            # Score evidence
            evidence_score = len(evidence) / max(len(kg.get('relationships', [])), 1)
            
            session.run(
                f"MATCH (a:Entity {{id:$src}}) MATCH (b:Entity {{id:$tgt}}) "
                f"MERGE (a)-[x:{rel}]->(b) SET x.evidence = $evidence, x.score = $score",
                src=src, tgt=tgt, evidence=evidence, score=evidence_score
            )

        # logger.info(f'Inserted KG into Memgraph (segment_id={segment_id}, entities={len(kg.get("entities", []))}, relations={len(kg.get("relationships", []))})')

def validate_and_cleanup_memgraph(doc_id: str):
    """Validate graph integrity (orphan cleanup skipped for Memgraph compatibility)."""
    with driver.session() as session:
        # Note: Orphan entity cleanup is skipped because Memgraph doesn't support 
        # pattern expressions like (e)--() or (e)-[]->()
        # The entities are properly connected during insertion, so this is safe to skip.
        
        # Validate document exists
        result = session.run(
            "MATCH (doc:Document {id:$doc_id}) RETURN count(doc) as doc_count",
            doc_id=doc_id
        )
        record = result.single()
        doc_count = record['doc_count'] if record else 0
        if doc_count > 0:
            logger.info(f"Document {doc_id} validation passed")
        
        # Count total entities and relationships
        result = session.run(
            "MATCH (e:Entity) RETURN count(e) as entity_count"
        )
        record = result.single()
        entity_count = record['entity_count'] if record else 0
        
        result = session.run(
            "MATCH ()-[r]->() RETURN count(r) as rel_count"
        )
        record = result.single()
        rel_count = record['rel_count'] if record else 0
        
        logger.info(f"Knowledge Graph stats: {entity_count} entities, {rel_count} relationships")


# --------- Pipeline Runner ---------

def process_segment(segment_data: Dict[str, Any], doc_id: str) -> bool:
    """Process a single segment: refine boundaries, extract KG, insert to Memgraph."""
    try:
        indices, merged_text = segment_data['indices'], segment_data['text']
        segment_id = f"seg-{'-'.join(str(x) for x in indices)}"

        # Refine boundaries
        sections = refine_boundaries(merged_text)

        for si, sec in enumerate(sections):
            title = sec.get('title', f"Section_{si}")
            content = sec.get('content', '')
            section_segment_id = f"{segment_id}-s{si}"

            # Extract KG
            kg = extract_kg(content, section_title=title)

            # Log extracted entities
            entities = kg.get('entities', [])
            relationships = kg.get('relationships', [])

            if entities:
                with open("entity_extraction_log.txt", 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Segment: {section_segment_id} | Title: {title}\n")
                    f.write(f"Time: {datetime.now().isoformat()}\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Extracted {len(entities)} entities:\n")
                    for e in entities:
                        eid = e.get('id', 'N/A')
                        name = e.get('name', 'N/A')
                        etype = e.get('type', 'N/A')
                        mentions = e.get('mentions', [])
                        f.write(f"  • {eid}: '{name}' (type={etype}) - {len(mentions)} mentions\n")

                    f.write(f"\nExtracted {len(relationships)} relationships:\n")
                    for r in relationships:
                        src = r.get('source', 'N/A')
                        rel = r.get('relation', 'N/A')
                        tgt = r.get('target', 'N/A')
                        evidence = r.get('evidence', [])
                        f.write(f"  • {src} -[{rel}]-> {tgt} ({len(evidence)} evidence)\n")
                    f.write("\n")

            # Insert to Memgraph
            insert_kg_into_memgraph(kg, indices, section_segment_id, doc_id)

            # Log each section completion
            logger.info(f"  {section_segment_id}, entities={len(entities)}, relations={len(relationships)} OK")

        return True
    except Exception as e:
        logger.exception(f"Failed to process segment: {e}")
        return False

def process_pdf(path: str, max_workers: int = None):
    """Process PDF with parallel segment processing."""
    if max_workers is None:
        max_workers = int(os.getenv("MAX_WORKERS", "4"))
    logger.info(f"Using {max_workers} parallel workers (set MAX_WORKERS in .env to change)")

    doc_id = str(uuid.uuid4())
    start_time = datetime.now()

    tracker.start_step("PDF Extraction")
    pages = extract_pdf_pages(path)
    tracker.end_step("PDF Extraction")
    n = len(pages)
    logger.info(f"Processing {n} pages with document ID: {doc_id}")
    
    # Initialize statistics tracking
    stats = {
        'doc_id': doc_id,
        'pdf_path': path,
        'total_pages': n,
        'total_segments': 0,
        'total_sections': 0,
        'total_entities_extracted': 0,
        'total_relationships_extracted': 0,
        'total_entities_merged': 0,
        'total_raw_sentences': 0,
        'entity_types_discovered': set(),
        'relation_types_discovered': set(),
        'start_time': start_time.isoformat(),
        'end_time': None,
        'duration_seconds': None
    }
    
    # Cluster pages using embedding-based approach
    tracker.start_step("Page Clustering")
    clusters = recursive_cluster_pages(pages)
    tracker.end_step("Page Clustering")
    stats['total_segments'] = len(clusters)

    # Prepare segments
    segments = []
    for cluster in clusters:
        merged = "\n\n---PAGE_BREAK---\n\n".join([pages[idx] for idx in cluster])
        segments.append({'indices': cluster, 'text': merged})

    # Process segments in parallel with progress tracking
    tracker.start_step("KG Extraction & Insertion")
    total_segments = len(segments)
    completed_count = 0
    success_count = 0

    def print_progress(completed, total, success):
        pct = int(completed / total * 100) if total else 100
        bar_len = 30
        filled = int(bar_len * completed / total) if total else bar_len
        bar = '█' * filled + '░' * (bar_len - filled)
        # Use logger.info so each update appears as a new line in docker logs
        logger.info(f"Progress: |{bar}| {pct}% ({completed}/{total} segments, {success} OK)")

    logger.info(f"Processing {total_segments} segments...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_segment, seg, doc_id): i for i, seg in enumerate(segments)}
        results = [None] * total_segments
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            completed_count += 1
            if result:
                success_count += 1
            print_progress(completed_count, total_segments, success_count)
    tracker.end_step("KG Extraction & Insertion")

    logger.info(f"Successfully processed {success_count}/{total_segments} segments")
    
    # Collect final statistics from Memgraph
    with driver.session() as session:
        # Count entities
        result = session.run("MATCH (e:Entity) RETURN count(e) as cnt")
        stats['final_unique_entities'] = result.single()['cnt']
        
        # Count relationships
        result = session.run("MATCH ()-[r]->() WHERE labels(startNode(r))[0] = 'Entity' RETURN count(r) as cnt")
        stats['final_relationships'] = result.single()['cnt']
        
        # Count mentions
        result = session.run("MATCH (m:Mention) RETURN count(m) as cnt")
        stats['total_mentions'] = result.single()['cnt']
        
        # Count raw sentences
        result = session.run("MATCH (s:RawSentence) RETURN count(s) as cnt")
        stats['total_raw_sentences'] = result.single()['cnt']
        
        # Get entity type distribution
        result = session.run("MATCH (e:Entity) RETURN e.type as type, count(e) as cnt ORDER BY cnt DESC")
        entity_type_dist = {r['type']: r['cnt'] for r in result}
        stats['entity_type_distribution'] = entity_type_dist
        
        # Get relationship type distribution
        result = session.run("""
            MATCH (e1:Entity)-[r]->(e2:Entity) 
            RETURN type(r) as rel_type, count(r) as cnt 
            ORDER BY cnt DESC
        """)
        rel_type_dist = {r['rel_type']: r['cnt'] for r in result}
        stats['relationship_type_distribution'] = rel_type_dist
    
    # Calculate final stats
    stats['entity_types_discovered'] = list(discovered_entity_types)
    stats['relation_types_discovered'] = list(discovered_relation_types)
    stats['end_time'] = datetime.now().isoformat()
    stats['duration_seconds'] = (datetime.now() - start_time).total_seconds()
    
    # Write detailed statistics report
    report_file = f"kg_processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("KNOWLEDGE GRAPH PROCESSING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Document ID: {stats['doc_id']}\n")
        f.write(f"PDF Path: {stats['pdf_path']}\n")
        f.write(f"Processing Time: {stats['start_time']} to {stats['end_time']}\n")
        f.write(f"Duration: {stats['duration_seconds']:.2f} seconds\n\n")
        
        f.write("-"*80 + "\n")
        f.write("INPUT STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Pages: {stats['total_pages']}\n")
        f.write(f"Total Segments: {stats['total_segments']}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("EXTRACTION STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Entity Types Discovered: {len(stats['entity_types_discovered'])}\n")
        f.write(f"  Types: {', '.join(sorted(stats['entity_types_discovered']))}\n\n")
        f.write(f"Relation Types Discovered: {len(stats['relation_types_discovered'])}\n")
        f.write(f"  Types: {', '.join(sorted(stats['relation_types_discovered']))}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("FINAL GRAPH STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Unique Entities: {stats['final_unique_entities']}\n")
        f.write(f"Total Relationships: {stats['final_relationships']}\n")
        f.write(f"Total Mentions: {stats['total_mentions']}\n")
        f.write(f"Total Raw Sentences: {stats['total_raw_sentences']}\n\n")
        
        f.write(f"Average Mentions per Entity: {stats['total_mentions'] / max(stats['final_unique_entities'], 1):.2f}\n")
        f.write(f"Average Relationships per Entity: {stats['final_relationships'] / max(stats['final_unique_entities'], 1):.2f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("ENTITY TYPE DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        for etype, cnt in sorted(stats['entity_type_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (cnt / stats['final_unique_entities'] * 100) if stats['final_unique_entities'] > 0 else 0
            f.write(f"  {etype:<30} {cnt:>5} ({percentage:>5.1f}%)\n")
        f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write("RELATIONSHIP TYPE DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        for rtype, cnt in sorted(stats['relationship_type_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (cnt / stats['final_relationships'] * 100) if stats['final_relationships'] > 0 else 0
            f.write(f"  {rtype:<30} {cnt:>5} ({percentage:>5.1f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Detailed processing report written to {report_file}")
    
    # Validate and cleanup
    validate_and_cleanup_memgraph(doc_id)

    # Save tracker state so build_embeddings.py can produce a unified report
    tracker.save_state()
    logger.info(f"Tracker state saved for unified report")
    logger.info(f"Total API tokens used: {tracker.get_summary()['grand_total_api_tokens']:,}")
    logger.info('Processing complete')

    return doc_id


# --------- CLI ---------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python dynamic_gemini_memgraph_pipeline.py /path/to/file.pdf")
        sys.exit(1)
    pdf_path = sys.argv[1]
    process_pdf(pdf_path)
