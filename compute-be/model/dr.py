from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import yaml
import re
from typing import List, Dict, Any, Tuple
import spacy
from spacy.tokens import Doc, Span

# Load spaCy model for entity detection
nlp = spacy.load("en_ner_bionlp13cg_md")
uri = "neo4j://localhost:7687"
username = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(username, password))

# def create_scibert_nlp():
#     """
#     Create a spaCy pipeline with SciBERT transformer.
#     """
#     config = {
#         "model": {
#             "@architectures": "spacy-transformers.TransformerModel.v3",
#             "name": "allenai/scibert_scivocab_uncased",
#             "tokenizer_config": {"use_fast": True},
#             "transformer_config": {}
#         }
#     }

#     nlp = spacy.blank("en")
    
#     # Add the transformer with simplified config
#     nlp.add_pipe("transformer", config=config)
#     nlp.add_pipe("sentencizer")
    
#     # Add entity ruler for medical terms
#     ruler = nlp.add_pipe("entity_ruler", config={"validate": True})
    
#     # Add patterns for common medical terms
#     patterns = [
#         {"label": "DISEASE", "pattern": "crohn's disease"},
#         {"label": "DISEASE", "pattern": "crohns disease"},
#         {"label": "DISEASE", "pattern": [{"LOWER": "crohn"}, {"LOWER": "'s"}, {"LOWER": "disease"}]},
#         {"label": "DISEASE", "pattern": [{"LOWER": "crohn"}, {"LOWER": "disease"}]},
#         {"label": "DISEASE", "pattern": "diabetes"},
#         {"label": "CHEMICAL", "pattern": [{"LOWER": "tnf"}, {"LOWER": "-"}, {"LOWER": "alpha"}]},
#     ]
#     ruler.add_patterns(patterns)
    
#     return nlp

# nlp = spacy.load("en_core_med7_lg")

class MedicalQueryPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with spaCy model and custom patterns"""
        # self.nlp = spacy.load("en_core_sci_scibert")
        
        # Add custom patterns for common medical terms and prefixes
        ruler = nlp.get_pipe("entity_ruler") if "entity_ruler" in nlp.pipe_names else nlp.add_pipe("entity_ruler")
        
        # Patterns for question prefixes and non-medical entities
        patterns = [
            {"label": "QUESTION_PREFIX", "pattern": [{"LOWER": {"IN": ["tell", "explain", "describe", "what", "how"]}}]},
            {"label": "QUESTION_PREFIX", "pattern": [{"LOWER": "what"}, {"LOWER": "is"}]},
            {"label": "QUESTION_PREFIX", "pattern": [{"LOWER": "tell"}, {"LOWER": "me"}, {"LOWER": "about"}]},
        ]
        ruler.add_patterns(patterns)

        # try:
        #     # Try to load scibert
        #     self.nlp = create_scibert_nlp()
        #     print("Successfully loaded SciBERT model")
        # except Exception as e:
        #     print(f"Error loading SciBERT: {e}")
        #     print("Falling back to default scientific model...")
        #     # Fallback to standard scientific model
        #     self.nlp = spacy.load("en_core_sci_md")
        
        # Common medical word combinations that should stay together
        self.medical_compounds = {
            "crohn's disease", "alzheimer's disease", "parkinson's disease",
            "multiple sclerosis", "breast cancer", "lung cancer",
            "type 2 diabetes", "high blood pressure"
        }
        
        # Load custom medical stopwords
        self.medical_stopwords = {
            "tell", "me", "about", "what", "is", "are", "can", "you", "explain",
            "describe", "know", "understand", "mean", "definition", "define"
        }

        self.medical_labels = {'DISEASE', 'CHEMICAL', 'GENE', 'PROTEIN', 'DRUG', 
                         'MEDCOND', 'DIAGNOSIS', 'MEDPROC', 'ANATOMY', 'SYMPTOM', 'COMPOUND',
                         'METABOLITE', 'PATHWAY', 'BIOLOGICAL_PROCESS', 'PEPTIDE', 'TRANSCRIPT', 'TISSUE',
                         }

    def is_medical_entity(self, text: str, ent_label: str) -> bool:
        """
        Determine if an entity is likely to be medical based on its text and label.
        """
        # Common medical entity labels from spaCy's sci model
        # medical_labels = {'DISEASE', 'CHEMICAL', 'GENE', 'PROTEIN', 'DRUG', 
        #                  'MEDCOND', 'DIAGNOSIS', 'MEDPROC', 'ANATOMY', 'SYMPTOM'}
        
        # Check if it's a known medical compound term
        if text.lower() in self.medical_compounds:
            return True
            
        # Check if it has a medical label
        if ent_label in self.medical_labels:
            return True
            
        # Additional checks for potential medical terms
        if any(pattern in text.lower() for pattern in [
            "disease", "syndrome", "disorder", "cancer", "itis",
            "osis", "emia", "gene", "protein", "receptor"
        ]):
            return True
            
        return False
    
    def is_medical_label(self, label: str) -> bool:
        """
        Check if a label is a known medical entity type.
        """
        print("Label:", label)
        return any((label in item.lower() or item.lower() in label) for item in self.medical_labels)

    def merge_adjacent_entities(self, doc: Doc) -> List[Tuple[str, str]]:
        """
        Merge adjacent entities that might be part of the same medical term.
        """
        merged_entities = []
        related_labels = []
        print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
        i = 0
        while i < len(doc.ents):
            current_ent = doc.ents[i]
            
            # Look ahead for adjacent entities
            if i + 1 < len(doc.ents):
                next_ent = doc.ents[i + 1]
                # Check if entities are adjacent and could form a medical term
                if next_ent.start == current_ent.end or \
                   (next_ent.start == current_ent.end + 1 and 
                    doc[current_ent.end].text.lower() in {"'s", "of", "and"}):
                    if self.is_medical_label(current_ent.text.lower()):
                        related_labels.append(current_ent.text.lower())
                        i += 1
                        continue
                    combined_text = doc[current_ent.start:next_ent.end].text
                    if self.is_medical_entity(combined_text, current_ent.label_):
                        merged_entities.append((combined_text, "MERGED_MEDICAL"))
                        i += 2
                        continue
            
            # if self.is_medical_entity(current_ent.text, current_ent.label_):
            #     merged_entities.append((current_ent.text, current_ent.label_))

            if current_ent.label_ == "ENTITY":
                merged_entities.append((current_ent.text, current_ent.label_))
            i += 1
            
        return merged_entities

    def extract_query_focus(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract the main medical entities from the query.
        Returns a list of dictionaries containing the text and type of each relevant entity.
        """
        doc = nlp(text)
        print("Entities 1111 :", [(ent.text, ent.label_) for ent in doc.ents])
        # doc = nlp(transcription)
        
        # First, merge adjacent entities that might be part of the same term
        merged_entities = self.merge_adjacent_entities(doc)
        print("Entities 2222 :", merged_entities)
        
        # Filter out question prefixes and non-medical entities
        relevant_entities = []
        for ent_text, ent_label in merged_entities:
            if ent_text.lower() not in self.medical_stopwords and \
               self.is_medical_entity(ent_text, ent_label):
                relevant_entities.append({
                    "text": ent_text,
                    "type": ent_label,
                    "original_text": text
                })
        if not relevant_entities:
            for ent in doc.ents:
                relevant_entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "original_text": text
                })
        return relevant_entities

    def preprocess_query(self, text: str) -> List[str]:
        """
        Preprocess the query to extract relevant medical entities.
        Returns a list of processed query terms.
        """
        # Extract relevant medical entities
        print("Processing query:", text)
        entities = self.extract_query_focus(text)
        print("Entities post focus:", entities)
        
        if not entities:
            # If no medical entities found, return cleaned original text
            return [re.sub(r'\s+', ' ', text).strip()]
        
        # Return the relevant medical terms
        return [entity["text"] for entity in entities]

def preprocess_query(text: str) -> List[str]:
    """
    Wrapper function for the MedicalQueryPreprocessor
    """
    preprocessor = MedicalQueryPreprocessor()
    return preprocessor.preprocess_query(text)

def detect_entities_from_index(text: str) -> List[Dict[str, str]]:
    """
    Detect medical entities by querying the Neo4j fulltext index.
    """
    def run_query(tx):
        # Query to search the fulltext index and return matches with their labels
         
        # cypher_query = """
        # CALL db.index.fulltext.queryNodes($index_name, $search_text, {limit: 5}) 
        # YIELD node, score
        # WHERE score > 0.5  // Add minimum score threshold
        # RETURN DISTINCT {
        #     text: node.name,
        #     label: head(labels(node)),
        #     score: score
        # } as result
        # ORDER BY result.score DESC
        # """
        embedding = model.encode(text).tolist()
        cypher_query = """
        CALL db.index.vector.queryNodes($index, $k, $embedding) 
                YIELD node, score WHERE score > 0.8
        // Add minimum score threshold
        RETURN DISTINCT {
            text: node.name,
            label: head(labels(node)),
            score: score
        } as result
        ORDER BY result.score DESC
        """
        result = tx.run(
            cypher_query, 
            index = "AllEntities",
            keyword_index = "all_entities_index",
            embedding = embedding,
            k=5,
            text_query=text[0]
        )
        
        return [record["result"] for record in result]

    with driver.session() as session:
        results = session.read_transaction(run_query)
        
        # Filter and format results
        entities = []
        seen = set()  # To avoid duplicates
        for result in results:
            if result['text'] and result['label']:
                # Create a unique key based on text and label
                key = f"{result['text']}_{result['label']}"
                if key not in seen:
                    entities.append({
                        "text": result['text'],
                        "label": result['label'],
                        "score": result['score']
                    })
                    seen.add(key)
        
        return entities

def fetch_context(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced context fetching with better entity detection.
    """
    try:
        query = parameters.get("text", "")
        
        # If no query provided, return empty result
        if not query:
            return ""
        
        # Preprocess the query
        processed_query = preprocess_query(query)
        
        # First, try to detect entities using Neo4j index
        print("Processed query:", processed_query)
        detected_entities = detect_entities_from_index(query)
        
        if not detected_entities:
            print(f"No entities detected for query: {query}")
            return ""
        
        # Use the highest scoring entity
        best_match = detected_entities[0]
        print(f"Detected entity: {best_match['text']} ({best_match['label']}) with score {best_match['score']}")
        
        # Proceed with the hybrid search using the detected entity
        index = "AllEntities"
        keyword_index = "all_entities_index"
        
        results = hybrid_search(driver, best_match['text'], index, keyword_index)
        
        # Generate recommendations based on the detected entity
        recommendations = []
        if results:
            recommendations = generate_recommendations(results[0], query)
        
        print("Recommendations:", recommendations)
        # Format results and include entity detection info
        formatted_results = format_yaml_for_llm(results)
        formatted_results += "\nDetected Entities:\n"
        for entity in detected_entities[:3]:  # Show top 3 matches
            formatted_results += f"- {entity['text']} ({entity['label']}) - Score: {entity['score']:.2f}\n"
        
        # if recommendations:
        #     formatted_results += "\nRecommended follow-up questions:\n"
        #     for i, rec in enumerate(recommendations, 1):
        #         formatted_results += f"{i}. {rec}\n"
        
        return formatted_results, recommendations
        
    except Exception as e:
        print(f"Error in fetch_context: {str(e)}")
        return ""

def detect_medical_entities(text: str) -> List[Dict[str, str]]:
    """
    Detect medical entities in the text and assign probable labels.
    """
    doc = nlp(text)
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    entities = []
    for ent in doc.ents:
        label = map_entity_type(ent.label_)
        if label:
            entities.append({
                "text": ent.text,
                "label": label
            })
    return entities

def map_entity_type(spacy_label: str) -> str:
    """
    Map spaCy entity labels to our Neo4j node labels.
    """
    mapping = {
        "DISEASE": "Disease",
        "CHEMICAL": "Drug",
        "GENE": "Gene",
        "PROTEIN": "Protein",
        "COMPOUND": "Compound",
        "METABOLITE": "Metabolite"
    }
    return mapping.get(spacy_label, None)

model = SentenceTransformer('BAAI/bge-large-en-v1.5', trust_remote_code=True)
def hybrid_search(driver, query, index, keyword_index, k=1):
    
    embedding = model.encode(query).tolist()
        
    def run_query(tx):
        cypher_query = """
        
        CALL {
            CALL db.index.vector.queryNodes($index, $k, $embedding)
            YIELD node, score
            RETURN node, score
            UNION
            CALL db.index.fulltext.queryNodes($keyword_index, $text_query, {limit: $k})
            YIELD node, score
            RETURN node, score
        }
        WITH node, score
        ORDER BY score DESC
        LIMIT $k
        MATCH (n:Protein|Drug|Phenotype|Disease|Gene|Metabolite|Pathway|Biological_process|Peptide|Transcript|Compound|Tissue|Symptom) 
        WHERE elementId(n) = node.f_key
        WITH n, score

        // Find related nodes of specific types within 2 hops
        OPTIONAL MATCH (n)-[r1]-(m:Disease)
        WHERE n <> m
        WITH n, score, collect(DISTINCT {node: m, relationship: r1, type: 'Disease'}) AS diseases

        OPTIONAL MATCH (n)-[r2]-(p:Protein)
        WHERE n <> p
        WITH n, score, diseases, collect(DISTINCT {node: p, relationship: r2, type: 'Protein'}) AS proteins

        OPTIONAL MATCH (n)-[r3]-(d:Drug|Compound)
        WHERE n <> d
        WITH n, score, diseases, proteins, collect(DISTINCT {node: d, relationship: r3, type: 'Drug'}) AS drugs

        OPTIONAL MATCH (n)-[r4]-(m:Metabolite)
        WHERE n <> m
        WITH n, score, diseases, proteins, drugs, collect(DISTINCT {node: m, relationship: r4, type: 'Metabolite'}) AS metabolites

        OPTIONAL MATCH (n)-[r5]-(g:Gene)
        WHERE n <> g
        WITH n, score, diseases, proteins, drugs, metabolites, collect(DISTINCT {node: g, relationship: r5, type: 'Gene'}) AS genes

        RETURN {
            score: score,
            metadata: labels(n)[0],
            root: apoc.map.removeKey(CASE WHEN properties(n) IS NOT NULL THEN properties(n) ELSE {} END, 'embedding'),
            related_nodes: {
                diseases: [d IN diseases[..3] | {
                    id: elementId(d.node),
                    properties: apoc.map.removeKey(CASE WHEN properties(d.node) IS NOT NULL THEN properties(d.node) ELSE {} END, 'embedding'),
                    relationship: type(d.relationship)
                }],
                proteins: [p IN proteins[..3] | {
                    id: elementId(p.node),
                    properties: apoc.map.removeKey(CASE WHEN properties(p.node) IS NOT NULL THEN properties(p.node) ELSE {} END, 'embedding'),
                    relationship: type(p.relationship)
                }],
                drugs: [d IN drugs[..3] | {
                    id: elementId(d.node),
                    properties: apoc.map.removeKey(CASE WHEN properties(d.node) IS NOT NULL THEN properties(d.node) ELSE {} END, 'embedding'),
                    relationship: type(d.relationship)
                }],
                metabolites: [m IN metabolites[..3] | {
                    id: elementId(m.node),
                    properties: apoc.map.removeKey(CASE WHEN properties(m.node) IS NOT NULL THEN properties(m.node) ELSE {} END, 'embedding'),
                    relationship: type(m.relationship)
                }],
                genes: [g IN genes[..3] | {
                    id: elementId(g.node),
                    properties: apoc.map.removeKey(CASE WHEN properties(g.node) IS NOT NULL THEN properties(g.node) ELSE {} END, 'embedding'),
                    relationship: type(g.relationship)
                }]
            }
        } AS result

        """
        result = tx.run(cypher_query, 
                       index=index, 
                       keyword_index=keyword_index, 
                       k=k, 
                       embedding=embedding, 
                       text_query=query,
                       max_hops=2)
        return [record["result"] for record in result]

    with driver.session() as session:
        results = session.read_transaction(run_query)
    return results

def generate_recommendations(context: Dict[str, Any], original_query: str) -> List[str]:
    """
    Generate follow-up question recommendations based on the context and original query.
    """
    recommendations = []
    root_entity = context.get('root', {}).get('name', '')
    metadata = context.get('metadata', '')
    
    related_nodes = context.get('related_nodes', {})
    
    # Add type-specific recommendations
    if related_nodes.get('proteins'):
        recommendations.append(f"What are the proteins associated with {root_entity}?")
    
    if related_nodes.get('drugs'):
        if metadata == 'Disease':
            recommendations.append(f"What are the drugs used to treat {root_entity}?")
        else:
            recommendations.append(f"What drugs interact with {root_entity}?")
    
    if related_nodes.get('genes'):
        recommendations.append(f"What genes are involved in {root_entity}?")
    
    if related_nodes.get('metabolites'):
        recommendations.append(f"What metabolites are associated with {root_entity}?")
    
    if metadata == 'Disease':
        recommendations.extend([
            f"What are the symptoms of {root_entity}?",
            f"What are the risk factors for {root_entity}?",
            f"What are the common biomarkers for {root_entity}?"
        ])
    
    # Return top 3-5 most relevant recommendations
    return recommendations[:4]

# def fetch_context(parameters: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Enhanced context fetching function that includes entity detection and recommendations.
#     """
#     try:
#         query = parameters["text"]
#         label = parameters["label"]
        
#         # If no specific entities are provided, detect them from the query
#         if not query or not label:
#             detected_entities = detect_medical_entities(query)
#             if detected_entities:
#                 # Use the first detected entity
#                 query = detected_entities[0]["text"]
#                 label = detected_entities[0]["label"]
        
#         index = "AllEntities"
#         keyword_index = "all_entities_index"
        
#         # Fetch context with improved Neo4j query
#         results = hybrid_search(driver, query, index, keyword_index)
        
#         # Generate recommendations based on the context
#         recommendations = []
#         if results:
#             recommendations = generate_recommendations(results[0], query)
        
#         # Format results and include recommendations
#         formatted_results = format_yaml_for_llm(results)
#         formatted_results += "\nRecommended follow-up questions:\n"
#         for i, rec in enumerate(recommendations, 1):
#             formatted_results += f"{i}. {rec}\n"
        
#         return formatted_results
    
#     except Exception as e:
#         print(f"Error fetching context: {str(e)}")
#         return ""
    

# The format_yaml_for_llm function needs to be updated to match the new structure
def format_yaml_for_llm(results, max_related_nodes=5):
    formatted_yaml = "Search Results:\n"
    for i, result in enumerate(results, 1):
        try:
            formatted_yaml += f"Result {i}:\n"
            formatted_yaml += f"  Score: {result['score']}\n"
            formatted_yaml += f"  Type: {result['metadata']}\n"
            formatted_yaml += "  Root Node:\n"
            for key, value in result['root'].items():
                formatted_yaml += f"    {key}: {format_value(value)}\n"
            formatted_yaml += "  Related Nodes:\n"
            for node in result['related_nodes'][:max_related_nodes]:
                formatted_yaml += f"    - ID: {node['id']}\n"
                formatted_yaml += f"      Labels: {', '.join(node['labels'])}\n"
                formatted_yaml += "      Properties:\n"
                for key, value in node['properties'].items():
                    formatted_yaml += f"        {key}: {format_value(value)}\n"
                formatted_yaml += "      Relationship:\n"
                formatted_yaml += f"        Type: {node['relationship']['type']}\n"
                if node['relationship']['properties']:
                    formatted_yaml += "        Properties:\n"
                    for key, value in node['relationship']['properties'].items():
                        formatted_yaml += f"          {key}: {format_value(value)}\n"
            if len(result['related_nodes']) > max_related_nodes:
                formatted_yaml += f"    ... (and {len(result['related_nodes']) - max_related_nodes} more related nodes)\n"
            formatted_yaml += "\n"
        except Exception as e:
            formatted_yaml += f"  Error processing result {i}: {str(e)}\n\n"
    return formatted_yaml

def format_value(value):
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(map(str, value)) + ']'
    elif isinstance(value, dict):
        return '{' + ', '.join(f'{k}: {v}' for k, v in value.items()) + '}'
    else:
        return str(value)