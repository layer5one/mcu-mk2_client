# /home/taylo/Cognition/memory/mega_memory_manager.py
import chromadb
from chromadb.utils import embedding_functions
import logging

logger = logging.getLogger("cognition_client")

class MegaMemoryManager:
    def __init__(self, persist_directory: str = "./mega_memory_db"):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="mega_memory",
                embedding_function=self.embedding_function
            )
            logger.info(f"Mega Memory ChromaDB collection loaded/created from {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize Mega Memory ChromaDB: {e}")
            self.client = None
            self.collection = None

    def add_memory_batch(self, memories: list) -> bool:
        if not self.collection:
            logger.error("Mega Memory collection not initialized.")
            return False
        
        documents =
        metadatas =
        ids =

        for mem in memories:
            doc_id = str(hash(mem['document']))
            documents.append(mem['document'])
            metadatas.append(mem['metadata'])
            ids.append(doc_id)
        
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added batch of {len(documents)} memories to Mega Memory.")
            return True
        except Exception as e:
            logger.error(f"Failed to add memory batch: {e}")
            return False

    def query_memory(self, query_text: str, n_results: int = 10, device_id: str = None) -> list:
        if not self.collection:
            logger.error("Mega Memory collection not initialized.")
            return
            
        where_filter = {}
        if device_id:
            where_filter = {"device_id": device_id}

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter
            )
            return results.get('documents', [])
        except Exception as e:
            logger.error(f"Failed to query Mega Memory: {e}")
            return
