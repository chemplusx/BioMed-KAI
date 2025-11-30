from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
import re
import spacy
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc

from ...tools.base_tool import BaseTool
from ...config.settings import settings


class KnowledgeGraphSearchTool(BaseTool):
    """
    Advanced search medical knowledge graph in Neo4j with entity detection,
    hybrid search, and recommendation generation
    """
    
    # Vector index names - should match what's configured in Neo4j
    # Note: Vector index "AllEntries" exists but seems inaccessible via procedure call
    # Using fulltext search which works reliably
    VECTOR_INDEX_NAME = None  # Disabled - fulltext search is working well
    FULLTEXT_INDEX_NAME = "all_entities_index"  # Fulltext index for keyword search
    
    # Properties to exclude from results (large/not useful for display)
    EXCLUDE_PROPERTIES = ['embedding']
    
    # Node labels that contain relationships (not shadow/index nodes)
    MEDICAL_NODE_LABELS = [
        'Disease', 'Drug', 'Symptom', 'Gene', 'Protein', 'Pathway', 
        'Metabolite', 'Compound', 'Phenotype', 'Tissue', 'Biological_process',
        'Chromosome', 'Transcript', 'Complex', 'Food', 'Modification'
    ]
    
    # Labels to filter out (index/shadow nodes)
    GENERIC_LABELS = ['AllEntries', 'Node', 'Entity']
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="knowledge_graph_search",
            description="Search medical knowledge graph for entities and relationships with AI-powered entity detection",
            config=config
        )
        
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
        # Initialize NLP components
        self._init_nlp_components()
        
        # Initialize sentence transformer for embeddings
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5', trust_remote_code=True)
        
        # Medical entity labels for NLP detection
        self.medical_labels = {
            'DISEASE', 'CHEMICAL', 'GENE', 'PROTEIN', 'DRUG', 
            'MEDCOND', 'DIAGNOSIS', 'MEDPROC', 'ANATOMY', 'SYMPTOM', 'COMPOUND',
            'METABOLITE', 'PATHWAY', 'BIOLOGICAL_PROCESS', 'PEPTIDE', 'TRANSCRIPT', 'TISSUE'
        }
        
        # Medical stopwords
        self.medical_stopwords = {
            "tell", "me", "about", "what", "is", "are", "can", "you", "explain",
            "describe", "know", "understand", "mean", "definition", "define"
        }
        
        # Medical compound terms
        self.medical_compounds = {
            "crohn's disease", "alzheimer's disease", "parkinson's disease",
            "multiple sclerosis", "breast cancer", "lung cancer",
            "type 2 diabetes", "high blood pressure"
        }
        
    def _init_nlp_components(self):
        """Initialize spaCy NLP model with medical entity recognition"""
        try:
            self.nlp = spacy.load("en_core_sci_md")
        except OSError:
            # Fallback to base English model if medical model not available
            self.nlp = spacy.load("en_core_web_sm")
            
        # Add entity ruler for medical terms
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", config={"validate": True})
            
            # Add patterns for common medical terms and question prefixes
            patterns = [
                {"label": "QUESTION_PREFIX", "pattern": [{"LOWER": {"IN": ["tell", "explain", "describe", "what", "how"]}}]},
                {"label": "QUESTION_PREFIX", "pattern": [{"LOWER": "what"}, {"LOWER": "is"}]},
                {"label": "QUESTION_PREFIX", "pattern": [{"LOWER": "tell"}, {"LOWER": "me"}, {"LOWER": "about"}]},
                {"label": "DISEASE", "pattern": "crohn's disease"},
                {"label": "DISEASE", "pattern": "crohns disease"},
                {"label": "DISEASE", "pattern": [{"LOWER": "crohn"}, {"LOWER": "'s"}, {"LOWER": "disease"}]},
                {"label": "DISEASE", "pattern": [{"LOWER": "crohn"}, {"LOWER": "disease"}]},
                {"label": "DISEASE", "pattern": "diabetes"},
                {"label": "CHEMICAL", "pattern": [{"LOWER": "tnf"}, {"LOWER": "-"}, {"LOWER": "alpha"}]},
            ]
            ruler.add_patterns(patterns)
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute enhanced knowledge graph search with entity detection"""
        query = kwargs.get("query", "")
        entity_types = kwargs.get("entity_types", ["Disease", "Drug", "Symptom", "Gene", "Protein"])
        limit = kwargs.get("limit", 5)
        include_relationships = kwargs.get("include_relationships", True)
        use_hybrid_search = kwargs.get("use_hybrid_search", True)
        generate_recommendations = kwargs.get("generate_recommendations", True)
        
        if not query:
            return {
                "query": query,
                "entities": [],
                "relationships": [],
                "context": "",
                "recommendations": [],
                "entity_count": 0,
                "relationship_count": 0,
                "error": "No query provided"
            }
        
        try:
            # Preprocess query and detect entities
            processed_entities = self._preprocess_query(query)
            print(f"[KG Search] Processing query: {query[:100]}...")
            print(f"[KG Search] Preprocessed entities: {processed_entities}")
            
            # Try vector search if index is configured
            detected_entities = []
            if self.VECTOR_INDEX_NAME:
                detected_entities = await self._detect_entities_from_index(query)
                print(f"[KG Search] Detected {len(detected_entities)} entities from vector index")
            else:
                print(f"[KG Search] Vector index disabled, using fulltext search only")
            
            # Determine search strategy
            if use_hybrid_search and detected_entities and self.VECTOR_INDEX_NAME:
                best_match = detected_entities[0]
                print(f"[KG Search] Using hybrid search with best match: {best_match.get('text', 'unknown')}")
                search_results = await self._hybrid_search(
                    best_match['text'],
                    k=limit,
                    entity_types=entity_types,
                )
                entities = self._format_hybrid_results(search_results)
            else:
                print(f"[KG Search] Using traditional fulltext search")
                entities = await self._search_entities(query, entity_types, limit)
            
            # Fallback: if no entities found, try fulltext search with the original query
            if not entities and detected_entities:
                print(f"[KG Search] No entities from hybrid search, trying fulltext fallback")
                entities = await self._search_entities(query, entity_types, limit)
            
            print(f"[KG Search] Found {len(entities)} entities")
            
            # Get relationships if requested
            relationships = []
            if include_relationships and entities:
                relationships = await self._get_relationships(entities[:3])
                print(f"[KG Search] Found {len(relationships)} relationships")
            
            # Generate recommendations
            recommendations = []
            if generate_recommendations and entities:
                recommendations = self._generate_recommendations(entities[0] if entities else {}, query)
            
            # Generate context
            context = self._generate_enhanced_context(entities, relationships, detected_entities)
            
            return {
                "query": query,
                "processed_entities": processed_entities,
                "detected_entities": detected_entities,
                "entities": entities,
                "relationships": relationships,
                "context": context,
                "recommendations": recommendations,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "search_method": "hybrid" if use_hybrid_search and detected_entities else "traditional"
            }
            
        except Exception as e:
            import traceback
            print(f"[KG Search] ERROR: {str(e)}")
            print(f"[KG Search] Traceback: {traceback.format_exc()}")
            return {
                "query": query,
                "entities": [],
                "relationships": [],
                "context": "",
                "recommendations": [],
                "entity_count": 0,
                "relationship_count": 0,
                "error": str(e)
            }
        
    def validate_params(self, **kwargs) -> bool:
        """Validate search parameters"""
        return bool(kwargs.get("query"))
    
    def _preprocess_query(self, text: str) -> List[str]:
        """Preprocess query to extract relevant medical entities"""
        entities = self._extract_query_focus(text)
        
        if not entities:
            return [re.sub(r'\s+', ' ', text).strip()]
        
        return [entity["text"] for entity in entities]
    
    def _extract_query_focus(self, text: str) -> List[Dict[str, Any]]:
        """Extract main medical entities from query"""
        doc = self.nlp(text)
        
        # Merge adjacent entities
        merged_entities = self._merge_adjacent_entities(doc)
        
        # Filter relevant entities
        relevant_entities = []
        for ent_text, ent_label in merged_entities:
            if (ent_text.lower() not in self.medical_stopwords and 
                self._is_medical_entity(ent_text, ent_label)):
                relevant_entities.append({
                    "text": ent_text,
                    "type": ent_label,
                    "original_text": text
                })
        
        # Fallback to all entities if no medical ones found
        if not relevant_entities:
            for ent in doc.ents:
                relevant_entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "original_text": text
                })
                
        return relevant_entities
    
    def _merge_adjacent_entities(self, doc: Doc) -> List[Tuple[str, str]]:
        """Merge adjacent entities that might be part of the same medical term"""
        merged_entities = []
        i = 0
        
        while i < len(doc.ents):
            current_ent = doc.ents[i]
            
            # Look for adjacent entities
            if i + 1 < len(doc.ents):
                next_ent = doc.ents[i + 1]
                
                # Check if entities are adjacent and could form a medical term
                if (next_ent.start == current_ent.end or 
                    (next_ent.start == current_ent.end + 1 and 
                     doc[current_ent.end].text.lower() in {"'s", "of", "and"})):
                    
                    combined_text = doc[current_ent.start:next_ent.end].text
                    if self._is_medical_entity(combined_text, current_ent.label_):
                        merged_entities.append((combined_text, "MERGED_MEDICAL"))
                        i += 2
                        continue
            
            if current_ent.label_ == "ENTITY" or self._is_medical_entity(current_ent.text, current_ent.label_):
                merged_entities.append((current_ent.text, current_ent.label_))
            i += 1
            
        return merged_entities
    
    def _is_medical_entity(self, text: str, ent_label: str) -> bool:
        """Determine if entity is medical based on text and label"""
        if text.lower() in self.medical_compounds:
            return True
            
        if ent_label in self.medical_labels:
            return True
            
        # Check for medical patterns
        medical_patterns = [
            "disease", "syndrome", "disorder", "cancer", "itis",
            "osis", "emia", "gene", "protein", "receptor"
        ]
        if any(pattern in text.lower() for pattern in medical_patterns):
            return True
            
        return False
    
    async def _detect_entities_from_index(self, text: str) -> List[Dict[str, Any]]:
        """Detect medical entities by querying Neo4j vector index
        
        Returns entities with their original type (from o_label or node labels)
        """
        try:
            print(f"[KG Search] Encoding text for vector search...")
            embedding = self.model.encode(text).tolist()
            print(f"[KG Search] Embedding generated, dimension: {len(embedding)}")
            
            def run_query(tx):
                # Query the vector index and get node properties including original label
                # Using low threshold to get more results
                cypher_query = """
                CALL db.index.vector.queryNodes($index, $k, $embedding) 
                YIELD node, score 
                RETURN DISTINCT {
                    text: node.name,
                    labels: labels(node),
                    o_label: node.o_label,
                    score: score,
                    f_key: node.f_key,
                    id: node.id
                } as result
                ORDER BY result.score DESC
                """
                result = tx.run(
                    cypher_query,
                    index=self.VECTOR_INDEX_NAME,
                    embedding=embedding,
                    k=15  # Get more results for better coverage
                )
                return [record["result"] for record in result]
            
            print(f"[KG Search] Querying vector index: {self.VECTOR_INDEX_NAME}")
            with self.driver.session() as session:
                results = session.read_transaction(run_query)
                print(f"[KG Search] Vector query returned {len(results)} raw results")
                
                entities = []
                seen = set()
                for result in results:
                    if result.get('text'):
                        # Get the original type - prefer o_label, then filter labels for non-generic ones
                        node_labels = result.get('labels', [])
                        specific_labels = [l for l in node_labels if l not in self.GENERIC_LABELS]
                        
                        original_type = (
                            result.get('o_label') or 
                            (specific_labels[0] if specific_labels else None) or
                            (node_labels[0] if node_labels else 'Unknown')
                        )
                        
                        # Create unique key based on name and type
                        key = f"{result['text'].lower()}_{original_type}"
                        if key not in seen:
                            entities.append({
                                "text": result['text'],
                                "type": original_type,
                                "label": original_type,  # Keep for backward compatibility
                                "o_label": result.get('o_label', original_type),
                                "score": result['score'],
                                "f_key": result.get('f_key', key),
                                "id": result.get('id')
                            })
                            seen.add(key)
                
                if entities:
                    print(f"[KG Search] Top entity: {entities[0].get('text')} ({entities[0].get('type')}) score: {entities[0].get('score')}")
                
                return entities
                
        except Exception as e:
            import traceback
            print(f"[KG Search] ERROR in _detect_entities_from_index: {str(e)}")
            print(f"[KG Search] Traceback: {traceback.format_exc()}")
            return []
    
    async def _hybrid_search(
        self,
        query: str,
        k: int = 5,
        entity_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid vector + fulltext search, optionally filtered by entity_types."""
        embedding = self.model.encode(query).tolist()
        entity_types = entity_types or []

        def run_query(tx):
            cypher = """
            CALL db.index.vector.queryNodes($index, $k, $embedding)
            YIELD node, score
            """
            if entity_types:
                cypher += """
                WHERE any(label IN labels(node) WHERE label IN $entity_types)
                   OR node.o_label IN $entity_types
                """
            cypher += """
            RETURN node AS root, score, labels(node) AS node_labels
            ORDER BY score DESC
            """

            params = {
                "index": self.VECTOR_INDEX_NAME,
                "k": k,
                "embedding": embedding,
                "entity_types": entity_types,
            }
            result = tx.run(cypher, **params)
            return [r.data() for r in result]

        with self.driver.session() as session:
            results = session.read_transaction(run_query)

        return results

    
    def _format_hybrid_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format hybrid search results to standard entity format"""
        entities = []
        for result in results:
            root = result.get('root', {})
            # Get node labels from the query result, filter out generic labels
            node_labels = result.get('node_labels', [])
            # Filter out 'AllEntries' and similar generic labels, prioritize specific ones
            specific_labels = [l for l in node_labels if l not in self.GENERIC_LABELS]
            # Use o_label from node properties if available, otherwise use first specific label
            primary_label = root.get('o_label') or (specific_labels[0] if specific_labels else (node_labels[0] if node_labels else 'Unknown'))
            
            # Filter out large/unnecessary properties like embeddings
            filtered_props = {k: v for k, v in root.items() if k not in self.EXCLUDE_PROPERTIES}
            
            entities.append({
                "id": root.get('f_key', root.get('id', 'unknown')),
                "name": root.get('name', 'Unknown'),
                "labels": specific_labels if specific_labels else node_labels,
                "type": primary_label,  # Add explicit type field
                "score": result.get('score', 0),
                "properties": filtered_props,
                "related_nodes": result.get('related_nodes', {})
            })
        return entities
    
    def _extract_medical_terms(self, query: str) -> List[str]:
        """Extract key medical terms from a query for multi-term search"""
        # Use NLP to extract entities
        doc = self.nlp(query)
        
        # Collect medical terms
        terms = set()
        
        # Get entities from NLP
        for ent in doc.ents:
            if ent.label_ not in ['QUESTION_PREFIX'] and len(ent.text) > 2:
                terms.add(ent.text.lower())
        
        # Also extract noun chunks that might be medical terms
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower().strip()
            # Check if it looks like a medical term
            if any(pattern in chunk_text for pattern in ['disease', 'syndrome', 'condition', 'itis', 'osis', 'emia']):
                terms.add(chunk_text)
            # Check against known medical patterns
            if self._is_medical_entity(chunk_text, ""):
                terms.add(chunk_text)
        
        # Fallback: extract individual words that might be medical terms
        medical_keywords = ['hypertension', 'asthma', 'diabetes', 'cancer', 'heart', 'lung', 
                          'kidney', 'liver', 'brain', 'stroke', 'infection', 'pain',
                          'fever', 'cough', 'fatigue', 'nausea', 'vertigo', 'dizziness']
        for word in query.lower().split():
            word = word.strip('.,?!')
            if word in medical_keywords or len(word) > 4 and any(p in word for p in ['tion', 'itis', 'osis', 'emia', 'pathy']):
                terms.add(word)
        
        return list(terms) if terms else [query]
    
    async def _search_entities(self, 
                              query: str,
                              entity_types: List[str],
                              limit: int) -> List[Dict[str, Any]]:
        """Traditional entity search using fulltext index with multi-term support"""
        entities = []
        seen_ids = set()
        
        try:
            # Extract medical terms from the query
            search_terms = self._extract_medical_terms(query)
            print(f"[KG Search] Extracted medical terms: {search_terms}")
            
            with self.driver.session() as session:
                # Search for each term to ensure we capture all relevant entities
                for term in search_terms[:5]:  # Limit to 5 terms
                    print(f"[KG Search] Searching for term: '{term}'")
                    
                    # Escape special Lucene characters and add fuzzy matching for typos
                    escaped_term = term.replace(':', '\\:').replace('*', '\\*').replace('?', '\\?')
                    # Add fuzzy matching (~) for terms > 4 chars to handle typos
                    if len(term) > 4 and ' ' not in term:
                        escaped_term = f"{escaped_term}~"  # Lucene fuzzy search
                    
                    cypher_query = """
                    CALL db.index.fulltext.queryNodes($index, $search_query) 
                    YIELD node, score
                    WHERE (any(label IN labels(node) WHERE label IN $entity_types)
                       OR node.o_label IN $entity_types)
                    RETURN node, score, labels(node) as labels
                    ORDER BY score DESC
                    LIMIT $limit
                    """
                    
                    result = session.run(
                        cypher_query,
                        index=self.FULLTEXT_INDEX_NAME,
                        search_query=escaped_term,
                        entity_types=entity_types,
                        limit=limit // len(search_terms) + 2  # Distribute limit across terms
                    )
                    
                    for record in result:
                        node = record["node"]
                        node_id = node.get("f_key", node.id)
                        
                        # Avoid duplicates
                        if node_id in seen_ids:
                            continue
                        seen_ids.add(node_id)
                        
                        node_labels = record["labels"]
                        specific_labels = [l for l in node_labels if l not in self.GENERIC_LABELS]
                        primary_label = node.get("o_label") or (specific_labels[0] if specific_labels else (node_labels[0] if node_labels else 'Unknown'))
                        
                        # Filter out large/unnecessary properties like embeddings
                        filtered_props = {k: v for k, v in dict(node).items() if k not in self.EXCLUDE_PROPERTIES}
                        
                        entities.append({
                            "id": node_id,
                            "name": node.get("name", "Unknown"),
                            "labels": specific_labels if specific_labels else node_labels,
                            "type": primary_label,
                            "score": record["score"],
                            "properties": filtered_props,
                            "matched_term": term  # Track which term matched
                        })
                
                # Sort by score
                entities.sort(key=lambda x: x.get('score', 0), reverse=True)
                entities = entities[:limit]  # Trim to limit
                
                print(f"[KG Search] Fulltext search returned {len(entities)} entities")
                    
        except Exception as e:
            import traceback
            print(f"[KG Search] ERROR in _search_entities: {str(e)}")
            print(f"[KG Search] Traceback: {traceback.format_exc()}")
                
        return entities
    
    async def _get_relationships(self, entities: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
        """Get relationships FROM entities to other relevant nodes (diseases, drugs, symptoms, etc.)
        
        This method handles the dual-node structure where:
        - AllEntries nodes are used for vector indexing with o_label pointing to original type
        - Original nodes (Disease, Symptom, etc.) hold the actual relationships
        
        Special focus on:
        - Drug contraindications (IS_A_CONTRAINDICATION_FOR)
        - Drug indications (INDICATED_FOR, IS_AN_INDICATION_FOR)
        - Drug interactions (INTERACTS_WITH)
        - Disease associations (ASSOCIATED_WITH)
        """
        relationships = []
        
        # Get entity names for name-based matching - extract core disease/drug names
        entity_names = []
        for e in entities:
            name = e.get("name", "").lower().strip()
            if name:
                # Also extract the core name without IDs (e.g., "hypertension" from "hypertension - 5466")
                core_name = name.split(' - ')[0].strip() if ' - ' in name else name
                entity_names.append(name)
                if core_name != name:
                    entity_names.append(core_name)
        
        entity_names = list(set(entity_names))  # Remove duplicates
        print(f"[KG Search] Searching relationships for: {entity_names[:5]}...")
        
        if not entity_names:
            return relationships
        
        with self.driver.session() as session:
            # Query relationships from nodes matching the entity names
            # Focus on clinically relevant relationships
            cypher_query = """
            // Find direct relationships
            MATCH (n)-[r]-(m)
            WHERE (toLower(n.name) IN $entity_names OR any(name IN $entity_names WHERE toLower(n.name) CONTAINS name))
            AND NOT 'AllEntries' IN labels(n)
            AND any(label IN labels(m) WHERE label IN $medical_labels)
            AND NOT 'AllEntries' IN labels(m)
            RETURN n, r, m, type(r) as rel_type, labels(n) as source_labels, labels(m) as target_labels
            ORDER BY 
                CASE type(r) 
                    WHEN 'IS_A_CONTRAINDICATION_FOR' THEN 1
                    WHEN 'INDICATED_FOR' THEN 2
                    WHEN 'IS_AN_INDICATION_FOR' THEN 3
                    WHEN 'INTERACTS_WITH' THEN 4
                    WHEN 'ASSOCIATED_WITH' THEN 5
                    ELSE 10
                END
            LIMIT $limit
            """
            
            result = session.run(
                cypher_query, 
                entity_names=entity_names, 
                limit=limit,
                medical_labels=self.MEDICAL_NODE_LABELS
            )
            
            seen_relationships = set()
            for record in result:
                source_node = record["n"]
                target_node = record["m"]
                rel_type = record["rel_type"]
                
                # Get meaningful labels (filter out generic ones)
                source_labels = [l for l in record["source_labels"] if l not in self.GENERIC_LABELS]
                target_labels = [l for l in record["target_labels"] if l not in self.GENERIC_LABELS]
                
                # Create unique key to avoid duplicates
                source_name = source_node.get('name', '')
                target_name = target_node.get('name', '')
                rel_key = f"{source_name}_{rel_type}_{target_name}"
                if rel_key in seen_relationships:
                    continue
                seen_relationships.add(rel_key)
                
                relationships.append({
                    "source": {
                        "id": source_node.get("id", source_node.get("f_key", "unknown")),
                        "name": source_name or "Unknown",
                        "type": source_labels[0] if source_labels else "Unknown"
                    },
                    "target": {
                        "id": target_node.get("id", target_node.get("f_key", "unknown")),
                        "name": target_name or "Unknown",
                        "type": target_labels[0] if target_labels else "Unknown"
                    },
                    "type": rel_type,
                    "properties": dict(record["r"])
                })
                
        return relationships
    
    def _generate_recommendations(self, context: Dict[str, Any], original_query: str) -> List[str]:
        """Generate follow-up question recommendations"""
        recommendations = []
        
        if not context:
            return recommendations
            
        root_name = context.get('name', context.get('properties', {}).get('name', ''))
        entity_type = context.get('type') or (context.get('labels', [''])[0] if context.get('labels') else '')
        related_nodes = context.get('related_nodes', {})
        
        # Type-specific recommendations
        if related_nodes.get('proteins'):
            recommendations.append(f"What are the proteins associated with {root_name}?")
        
        if related_nodes.get('drugs'):
            if entity_type == 'Disease':
                recommendations.append(f"What are the drugs used to treat {root_name}?")
            else:
                recommendations.append(f"What drugs interact with {root_name}?")
        
        if related_nodes.get('genes'):
            recommendations.append(f"What genes are involved in {root_name}?")
        
        if related_nodes.get('metabolites'):
            recommendations.append(f"What metabolites are associated with {root_name}?")
        
        if entity_type == 'Disease':
            recommendations.extend([
                f"What are the symptoms of {root_name}?",
                f"What are the risk factors for {root_name}?",
                f"What are the common biomarkers for {root_name}?"
            ])
        
        return recommendations[:4]
    
    def _generate_enhanced_context(self, 
                                 entities: List[Dict[str, Any]],
                                 relationships: List[Dict[str, Any]],
                                 detected_entities: List[Dict[str, Any]]) -> str:
        """Generate enhanced human-readable context"""
        context_parts = []
        
        # Detected entities section
        if detected_entities:
            context_parts.append("Detected Entities:")
            for entity in detected_entities[:3]:
                entity_type = entity.get('type') or entity.get('o_label') or entity.get('label', 'Unknown')
                context_parts.append(
                    f"- {entity['text']} ({entity_type}) - Score: {entity['score']:.2f}"
                )
            context_parts.append("")
        
        # Main entities section
        context_parts.append("Search Results:")
        for i, entity in enumerate(entities[:3], 1):
            entity_type = entity.get('type') or (entity["labels"][0] if entity.get("labels") else "Entity")
            name = entity.get("name", "Unknown")
            
            context_parts.append(f"Result {i}:")
            context_parts.append(f"  Type: {entity_type}")
            context_parts.append(f"  Name: {name}")
            context_parts.append(f"  Score: {entity.get('score', 'N/A')}")
            
            # Add key properties
            props = entity.get("properties", {})
            if "description" in props:
                context_parts.append(f"  Description: {props['description']}")
            if "synonyms" in props:
                context_parts.append(f"  Also known as: {props['synonyms']}")
            
            # Add related nodes information
            related_nodes = entity.get("related_nodes", {})
            if related_nodes:
                context_parts.append("  Related Entities:")
                for category, nodes in related_nodes.items():
                    if nodes:
                        context_parts.append(f"    {category.capitalize()}:")
                        for node in nodes[:2]:  # Limit to 2 per category
                            node_name = node.get('properties', {}).get('name', 'Unknown')
                            relationship = node.get('relationship', 'related_to')
                            context_parts.append(f"      - {node_name} ({relationship})")
            
            context_parts.append("")
        
        # Relationships section
        if relationships:
            context_parts.append("Direct Relationships:")
            for rel in relationships[:5]:
                source_type = rel['source'].get('type', '')
                target_type = rel['target'].get('type', '')
                context_parts.append(
                    f"- {rel['source']['name']} ({source_type}) --[{rel['type']}]--> {rel['target']['name']} ({target_type})"
                )
        
        return "\n".join(context_parts)
    
    def __del__(self):
        """Close Neo4j driver connection"""
        if hasattr(self, 'driver'):
            self.driver.close()

