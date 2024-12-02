import os
from py2neo import Graph
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class Neo4jAllEntriesBuilder:
    def __init__(self, uri, username, password):
        """
        Initialize Neo4j graph connection
        
        :param uri: Neo4j database URI
        :param username: Database username
        :param password: Database password
        """
        self.graph = Graph(uri, auth=(username, password))
    
    def get_all_labels(self):
        """
        Retrieve all existing labels in the database
        
        :return: List of label names (excluding system labels)
        """
        return ['Biological_process','Chromosome','Compound','Disease','Drug','Experimental_factor','Gene','GWAS_study','Metabolite','Molecular_function','Pathway','Peptide','Protein','Publication','Symptom','Tissue','Transcript']
        query = """
        CALL db.labels() YIELD label
        WHERE label <> 'AllEntries' AND label <> 'UNWIND'
        RETURN label
        """
        return [record['label'] for record in self.graph.run(query)]
    
    def build_all_entries(self):
        """
        Build AllEntries label across all existing labels
        Ensures no duplicate entries by using a unique constraint
        """
        # First, create a unique constraint on f_key to prevent duplicates
        create_constraint_query = """
        CREATE CONSTRAINT unique_all_entries_fkey 
        IF NOT EXISTS 
        FOR (a:AllEntries) 
        REQUIRE a.f_key IS UNIQUE
        """
        self.graph.run(create_constraint_query)
        
        # Get all labels
        labels = self.get_all_labels()
        print(f"Found {len(labels)} labels to process")
        
        # Delete existing AllEntries nodes
        delete_query = "MATCH (a:AllEntries) DETACH DELETE a"
        self.graph.run(delete_query)
        
        # Create AllEntries nodes with MERGE to prevent duplicates
        create_all_entries_query = """
        CALL apoc.periodic.iterate(
            "UNWIND $labels AS current_label
             MATCH (n)
             WHERE current_label IN labels(n)
             RETURN n, current_label",
            "MERGE (a:AllEntries {f_key: id(n)})
             ON CREATE SET 
                 a.name = COALESCE(n.name, '') + ' - ' + toString(id(n)),
                 a.o_label = current_label",
            {batchSize: 1000, parallel: true, params: {labels: $labels}}
        )
        """
        
        try:
            # Execute the comprehensive query
            self.graph.run(create_all_entries_query, {'labels': labels})
            
            # Verify the results
            count_query = "MATCH (a:AllEntries) RETURN count(a) AS total_entries"
            total_entries = self.graph.run(count_query).data()[0]['total_entries']
            
            print(f"Total unique entries created in AllEntries label: {total_entries}")
        
        except Exception as e:
            print(f"Error creating AllEntries nodes: {e}")
            raise

# Example usage
def main():
    # Configure your Neo4j connection details
    NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
    
    # Initialize the builder
    builder = Neo4jAllEntriesBuilder(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    
    # Build AllEntries
    builder.build_all_entries()

if __name__ == '__main__':
    main()