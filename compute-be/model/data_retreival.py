from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import yaml

model = SentenceTransformer('BAAI/bge-large-en-v1.5',trust_remote_code=True)  #('BAAI/bge-large-en-v1.5')
# dunzhang/stella_en_400M_v5
# dunzhang/stella_en_1.5B_v5
def hybrid_search(driver, query, index, keyword_index, k=1):
    print("Query: ", query, "Index: ", index, "Keyword Index: ", keyword_index, "K: ", k)
    # Load the embedding model
    # Generate embedding for the query
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
        MATCH (n:Protein|Drug|Phenotype|Disease|Gene|Metabolite|Pathway|Biological_process|Peptide|Transcript|Compound|Tissue|Symptom) WHERE elementId(n) = node.f_key
        WITH n, score
        CALL apoc.path.spanningTree(n, {
            maxLevel: $max_hops,
            labelFilter: '+Drug|Phenotype|Disease|Gene|Metabolite|Pathway|Biological_process|Compound',
            limit: 10
        }) YIELD path
        WITH n, score, relationships(path) AS rels, last(nodes(path)) AS related
        WHERE n <> related
        WITH n, score, collect(distinct {
            node: related,
            relationship: last(rels)
        }) AS related_info
        RETURN {
            score: score,
            metadata: labels(n)[0],
            root: apoc.map.removeKey(properties(n), 'embedding'),
            related_nodes: [related IN related_info | {
                id: id(related.node),
                labels: labels(related.node),
                properties: apoc.map.removeKey(properties(related.node), 'embedding'),
                relationship: {
                    type: type(related.relationship),
                    properties: properties(related.relationship)
                }
            }]
        } AS result
        """
        result = tx.run(cypher_query, 
                        index=index, 
                        keyword_index=keyword_index, 
                        k=k, 
                        embedding=embedding, 
                        text_query=query,
                        max_hops=1)
        return [record["result"] for record in result]

    with driver.session() as session:
        results = session.read_transaction(run_query)
    
    return results

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

# Usage
uri = "neo4j://localhost:7687"
username = "neo4j"
password = "password"

driver = GraphDatabase.driver(uri, auth=(username, password))


def fetch_context(parameters):
    print(parameters)
    try:
        query = parameters["text"]
        label = parameters["label"]
        index = "AllEntities"
        keyword_index = "all_entities_index"
        # results = hybrid_search(driver, query, index, keyword_index)
        
        # for result in results:
        #     print(f"Text: {result['text']}")
        #     print(f"Score: {result['score']}")
        #     print(f"Metadata: {result['metadata']}")
        #     print("---")
        print("Query:", query, "Label:", label, "Parameters:", parameters)
        yaml_results = hybrid_search(driver, query, index, keyword_index)
        print("YAML Results:", yaml_results)
        formatted_yaml = format_yaml_for_llm(yaml_results)
        
        print("YAML Output:")
        print(formatted_yaml)
        return formatted_yaml
    except Exception as e:
        print("Error fetching context:", e)
        return ""
    finally:
        driver.close()

    # return {
    #     "context": "This is a context",
    #     "parameters": parameters
    # }