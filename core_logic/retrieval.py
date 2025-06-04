from typing import List, Tuple, Optional
from rag_pipeline.embedding_generator import EmbeddingGenerator, Embedding
from rag_pipeline.vector_store_manager import VectorStoreManager, LangchainDocument

class DocumentRetriever:
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store_manager: VectorStoreManager):
        if not isinstance(embedding_generator, EmbeddingGenerator):
            raise TypeError("embedding_generator must be an instance of EmbeddingGenerator")
        if not isinstance(vector_store_manager, VectorStoreManager):
            raise TypeError("vector_store_manager must be an instance of VectorStoreManager")
            
        self.embedding_generator = embedding_generator
        self.vector_store_manager = vector_store_manager

    def retrieve_relevant_documents(self, query: str, top_k: int = 5) -> List[Tuple[LangchainDocument, float]]:
        if not query:
            print("Error: Query cannot be empty for retrieval.")
            return []

        print(f"\nRetrieving documents for query: '{query}'")
        
        query_embedding: Optional[Embedding] = self.embedding_generator.generate_single_embedding(query)
        
        if query_embedding is None:
            print("Error: Failed to generate embedding for the query.")
            return []

        retrieved_docs_with_scores = self.vector_store_manager.search(query_embedding, top_k=top_k)
        
        if not retrieved_docs_with_scores:
            print("No relevant documents found in the vector store for this query.")
            return []
            
        print(f"Retrieved {len(retrieved_docs_with_scores)} relevant document chunks.")
            
        return retrieved_docs_with_scores

if __name__ == '__main__':
    print("DocumentRetriever Example:")

    try:
        embed_gen = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Failed to initialize EmbeddingGenerator for example: {e}. Skipping example.")
        exit()
    
    dummy_index_path = "data/vector_store/retriever_test.index"
    dummy_metadata_path = "data/vector_store/retriever_test_meta.pkl"
    
    import os
    if os.path.exists(dummy_index_path): os.remove(dummy_index_path)
    if os.path.exists(dummy_metadata_path): os.remove(dummy_metadata_path)

    vector_store_mgr = VectorStoreManager(index_path=dummy_index_path, metadata_path=dummy_metadata_path)

    dummy_doc_texts = [
        "A firewall is a network security device that monitors and filters incoming and outgoing network traffic.",
        "SQL injection is a code injection technique used to attack data-driven applications.",
        "Cross-Site Scripting (XSS) attacks enable attackers to inject client-side scripts into web pages viewed by other users.",
        "A VPN, or Virtual Private Network, creates a secure, encrypted connection over a less secure network, such as the internet."
    ]
    dummy_doc_chunks_lc: List[LangchainDocument] = [
        {"page_content": text, "metadata": {"source": f"dummy_doc_{i+1}.txt"}} for i, text in enumerate(dummy_doc_texts)
    ]
    
    dummy_embeddings = embed_gen.generate_embeddings([d["page_content"] for d in dummy_doc_chunks_lc])

    if not dummy_embeddings:
        print("Failed to generate dummy embeddings for VectorStoreManager setup. Skipping example.")
        exit()
        
    try:
        vector_store_mgr.create_and_save_vector_store(dummy_doc_chunks_lc, dummy_embeddings)
        if not vector_store_mgr.load_vector_store():
            print("Failed to load the dummy vector store for the retriever. Skipping example.")
            exit()
    except Exception as e:
        print(f"Error setting up dummy vector store for retriever: {e}. Skipping example.")
        exit()

    retriever = DocumentRetriever(embedding_generator=embed_gen, vector_store_manager=vector_store_mgr)

    test_query = "What is SQL injection?"
    retrieved_results = retriever.retrieve_relevant_documents(test_query, top_k=2)

    if retrieved_results:
        print(f"\nTop {len(retrieved_results)} results for query: '{test_query}'")
        for i, (doc, score) in enumerate(retrieved_results):
            print(f"  Result {i+1}:")
            print(f"    Content: {doc['page_content']}")
            print(f"    Source: {doc['metadata']['source']}")
            print(f"    Score (Distance): {score:.4f}")
    else:
        print(f"No documents retrieved for query: '{test_query}'")