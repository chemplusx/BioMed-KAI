

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import logging
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class Neo4jEmbeddingGenerator:
    def __init__(self, uri, user, password, model_name="BAAI/bge-large-en-v1.5", 
                 batch_size=100, num_workers=2, queue_size=10):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Setup queues for producer-consumer pattern
        self.read_queue = queue.Queue(maxsize=queue_size)
        self.write_queue = queue.Queue(maxsize=queue_size)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Control flags
        self.stop_event = threading.Event()
        self.reading_complete = threading.Event()
        
        # Progress tracking
        self.total_processed = 0
        self.total_saved = 0

    def close(self):
        self.driver.close()

    def get_names_batch(self, skip):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.name is not null AND n.embedding is null
                RETURN elementId(n) as node_id, n.name as name
                ORDER BY elementId(n)
                SKIP $skip
                LIMIT $batch_size
                """,
                skip=skip,
                batch_size=self.batch_size
            )
            return [{"node_id": record["node_id"], "name": record["name"]} 
                   for record in result]

    def save_embeddings_batch(self, nodes_with_embeddings):
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $nodes as node
                MATCH (n)
                WHERE elementId(n) = node.node_id
                SET n.embedding = node.embedding
                RETURN count(*) as updated
                """,
                nodes=nodes_with_embeddings
            )
            return result.single()["updated"]

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [embedding.tolist() for embedding in embeddings]

    def reader_worker(self):
        """Worker that reads from Neo4j and puts batches into the read queue"""
        skip = 0
        try:
            while not self.stop_event.is_set():
                print("Reading batch")
                batch = self.get_names_batch(skip)
                if not batch:
                    break
                
                self.read_queue.put(batch)
                skip += self.batch_size
                
            self.reading_complete.set()
        except Exception as e:
            self.logger.error(f"Error in reader worker: {str(e)}")
            self.stop_event.set()

    def embedding_worker(self):
        """Worker that generates embeddings using GPU"""
        try:
            while not (self.reading_complete.is_set() and self.read_queue.empty()):
                try:
                    batch = self.read_queue.get(timeout=5)
                except queue.Empty:
                    if self.reading_complete.is_set():
                        break
                    continue
                print("Creating embeddings for batch", len(batch))
                names = [node["name"] for node in batch]
                embeddings = self.generate_embeddings(names)
                
                nodes_with_embeddings = [
                    {"node_id": node["node_id"], "embedding": embedding}
                    for node, embedding in zip(batch, embeddings)
                ]
                print("Created embeddings for batch", len(batch))
                self.write_queue.put(nodes_with_embeddings)
                self.total_processed += len(batch)
                self.read_queue.task_done()
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.logger.error(f"Error in embedding worker: {str(e)}")
            self.stop_event.set()

    def writer_worker(self):
        """Worker that saves embeddings back to Neo4j"""
        try:
            while not (self.reading_complete.is_set() and self.write_queue.empty() 
                      and self.read_queue.empty()):
                try:
                    batch = self.write_queue.get(timeout=5)
                except queue.Empty:
                    if self.reading_complete.is_set() and self.read_queue.empty():
                        break
                    continue
                print("Saving batch")
                updated = self.save_embeddings_batch(batch)
                self.total_saved += updated
                self.logger.info(f"Saved batch of {updated} nodes. "
                               f"Total saved: {self.total_saved}")
                self.write_queue.task_done()
                
        except Exception as e:
            self.logger.error(f"Error in writer worker: {str(e)}")
            self.stop_event.set()

    def process_all_nodes(self):
        """Process all nodes using parallel workers"""
        try:
            # Start reader thread
            reader_thread = threading.Thread(target=self.reader_worker)
            reader_thread.start()
            
            # Start embedding thread (GPU operations)
            embedding_thread = threading.Thread(target=self.embedding_worker)
            embedding_thread.start()
            
            # Start writer threads pool
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                writer_futures = [
                    executor.submit(self.writer_worker)
                    for _ in range(self.num_workers)
                ]
            
            # Wait for all threads to complete
            reader_thread.join()
            embedding_thread.join()
            
            if self.stop_event.is_set():
                raise Exception("Processing stopped due to an error")
            
            return self.total_saved
            
        except Exception as e:
            self.logger.error(f"Error in process_all_nodes: {str(e)}")
            self.stop_event.set()
            raise

def main():
    # Configuration
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "password"
    BATCH_SIZE = 100000
    NUM_WORKERS = 16
    QUEUE_SIZE = 10
    
    try:
        generator = Neo4jEmbeddingGenerator(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            queue_size=QUEUE_SIZE
        )
        
        total_processed = generator.process_all_nodes()
        print(f"Successfully processed {total_processed} nodes")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
    finally:
        generator.close()

if __name__ == "__main__":
    main()