import os
import pickle
from typing import List, Tuple, Dict, Union
import numpy as np
import faiss

Embedding = Union[List[float], np.ndarray]
LangchainDocument = Dict[str, Union[str, Dict]]

class VectorStoreManager:
    def __init__(self, index_path: str = "data/vector_store/faiss_index.index", 
                 metadata_path: str = "data/vector_store/faiss_index.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.document_chunks: List[LangchainDocument] = []

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    def create_and_save_vector_store(self, 
                                     doc_chunks: List[LangchainDocument], 
                                     embeddings: List[Embedding]):
        if not doc_chunks or not embeddings:
            raise ValueError("Document chunks and embeddings cannot be empty.")
        if len(doc_chunks) != len(embeddings):
            raise ValueError("The number of document chunks and embeddings must be the same.")

        embeddings_np = np.array(embeddings).astype('float32')
        
        dimension = embeddings_np.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        
        self.document_chunks = doc_chunks

        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            print(f"FAISS index saved to: {self.index_path}")
            print(f"Document chunks metadata saved to: {self.metadata_path}")
        except Exception as e:
            print(f"Error saving FAISS index or metadata: {e}")
            raise

    def load_vector_store(self) -> bool:
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            print("Vector store files not found. Please create the store first.")
            return False
        
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.document_chunks = pickle.load(f)
            print(f"FAISS index loaded from: {self.index_path}")
            print(f"Document chunks metadata loaded from: {self.metadata_path}")
            if self.index.ntotal == 0 or not self.document_chunks:
                print("Warning: Loaded FAISS index or document chunks are empty.")
            elif self.index.ntotal != len(self.document_chunks):
                print(f"Warning: Mismatch between FAISS index size ({self.index.ntotal}) and "
                      f"number of document chunks ({len(self.document_chunks)}).")

            return True
        except Exception as e:
            print(f"Error loading FAISS index or metadata: {e}")
            self.index = None
            self.document_chunks = []
            return False

    def search(self, query_embedding: Embedding, top_k: int = 5) -> List[Tuple[LangchainDocument, float]]:
        if self.index is None or not self.document_chunks:
            print("Error: Vector store (index or chunks) not loaded. Cannot perform search.")
            return []
        if self.index.ntotal == 0:
            print("Warning: FAISS index is empty. Cannot perform search.")
            return []

        query_embedding_np = np.array([query_embedding]).astype('float32')

        try:
            distances, indices = self.index.search(query_embedding_np, top_k)
            
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                if idx < 0 or idx >= len(self.document_chunks):
                    print(f"Warning: Retrieved invalid index {idx} from FAISS search.")
                    continue
                results.append((self.document_chunks[idx], float(dist)))
            
            return results
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return []

if __name__ == '__main__':
    print("VectorStoreManager Example:")

    test_index_path = "data/vector_store/test_faiss.index"
    test_metadata_path = "data/vector_store/test_faiss_meta.pkl"
    
    if os.path.exists(test_index_path): os.remove(test_index_path)
    if os.path.exists(test_metadata_path): os.remove(test_metadata_path)

    dummy_chunks: List[LangchainDocument] = [
        {"page_content": "Alpha document about firewalls.", "metadata": {"source": "doc1.txt", "id": "alpha"}},
        {"page_content": "Bravo document discussing VPNs.", "metadata": {"source": "doc2.txt", "id": "bravo"}},
        {"page_content": "Charlie article on malware.", "metadata": {"source": "doc3.md", "id": "charlie"}},
        {"page_content": "Delta piece about phishing attacks.", "metadata": {"source": "doc4.txt", "id": "delta"}},
    ]
    dummy_embeddings = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.1]
    ]).astype('float32')

    if len(dummy_chunks) != len(dummy_embeddings):
        print("Error in dummy data setup: Mismatch between chunks and embeddings count.")
    else:
        vector_store_mgr = VectorStoreManager(index_path=test_index_path, metadata_path=test_metadata_path)

        try:
            print("\nCreating and saving vector store...")
            vector_store_mgr.create_and_save_vector_store(dummy_chunks, dummy_embeddings.tolist())
        except ValueError as ve:
            print(f"ValueError during store creation: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during store creation: {e}")

        print("\nLoading vector store...")
        loaded_vector_store_mgr = VectorStoreManager(index_path=test_index_path, metadata_path=test_metadata_path)
        if loaded_vector_store_mgr.load_vector_store():
            print("Vector store loaded successfully for searching.")
            
            query_emb_firewall = np.array([0.15, 0.25, 0.35]).astype('float32')
            
            print(f"\nSearching for documents similar to query (simulating 'firewall')...")
            search_results = loaded_vector_store_mgr.search(query_emb_firewall.tolist(), top_k=2)
            
            if search_results:
                print("Search Results:")
                for doc, score in search_results:
                    print(f"  Content: {doc['page_content'][:50]}... (Score/Distance: {score:.4f}) (Source: {doc['metadata']['source']})")
            else:
                print("No search results found or error during search.")
        else:
            print("Failed to load vector store for searching.")