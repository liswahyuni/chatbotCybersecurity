from typing import Optional, Union, Iterator, Dict, Any, Tuple, List
from rag_pipeline.embedding_generator import EmbeddingGenerator
from rag_pipeline.vector_store_manager import VectorStoreManager
from rag_pipeline.llm_interface import LLMInterface
from rag_pipeline.conversation_memory import SimpleConversationMemory
from core_logic.retrieval import DocumentRetriever
from core_logic.generation import ResponseGenerator
import os
from dotenv import load_dotenv

load_dotenv()

class RagPipeline:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_index_path: str = "data/vector_store/faiss_index.index",
        vector_store_metadata_path: str = "data/vector_store/faiss_index.pkl",
        llm_model_name: str = "qwen:0.5b",
        ollama_host: Optional[str] = None,
        max_conversation_history: int = 6
    ):
        print("Initializing RAG Pipeline...")
        
        try:
            self.embedding_generator = EmbeddingGenerator(model_name=embedding_model_name)
            print("EmbeddingGenerator initialized.")
        except Exception as e:
            print(f"Failed to initialize EmbeddingGenerator: {e}")
            raise

        self.vector_store_manager = VectorStoreManager(
            index_path=vector_store_index_path,
            metadata_path=vector_store_metadata_path
        )
        if not self.vector_store_manager.load_vector_store():
            print("Error: Failed to load vector store. Ensure 'build_vector_store.py' has been run successfully.")
            print(f"Checked paths: Index='{vector_store_index_path}', Metadata='{vector_store_metadata_path}'")
            raise FileNotFoundError("Vector store not found or failed to load. Please build it first.")
        print("VectorStoreManager loaded.")

        self.document_retriever = DocumentRetriever(
            embedding_generator=self.embedding_generator,
            vector_store_manager=self.vector_store_manager
        )
        print("DocumentRetriever initialized.")

        try:
            self.llm_interface = LLMInterface(model_name=llm_model_name, ollama_host=ollama_host)
            print("LLMInterface initialized.")
        except Exception as e:
            print(f"Failed to initialize LLMInterface: {e}")
            print("Ensure Ollama is running and the model is pulled.")
            raise

        self.response_generator = ResponseGenerator(llm_interface=self.llm_interface)
        print("ResponseGenerator initialized.")
        
        self.conversation_memory = SimpleConversationMemory(max_history_length=max_conversation_history)
        print("SimpleConversationMemory initialized.")
        
        print("RAG Pipeline initialized successfully.")

    def process_query(
        self, 
        query: str, 
        stream: bool = False, 
        top_k_retrieval: int = 3,
        llm_options: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        
        if not query.strip():
            error_msg = "Error: Query cannot be empty."
            return error_msg, []

        retrieved_docs_with_scores = self.document_retriever.retrieve_relevant_documents(
            query,
            top_k=top_k_retrieval
        )
        
        context_texts_for_deepeval: List[str] = []
        if retrieved_docs_with_scores:
            for doc, score in retrieved_docs_with_scores:
                if isinstance(doc, dict) and 'page_content' in doc:
                    context_texts_for_deepeval.append(str(doc['page_content']))
                elif isinstance(doc, str):
                    context_texts_for_deepeval.append(doc)

        current_history = self.conversation_memory.get_history()

        if llm_options is None:
            llm_options = {}

        llm_response_or_stream = self.response_generator.get_augmented_response(
            user_query=query,
            retrieved_docs_with_scores=retrieved_docs_with_scores,
            conversation_history=current_history,
            stream=stream,
            llm_options=llm_options
        )

        if not stream and isinstance(llm_response_or_stream, str) and not llm_response_or_stream.startswith("Error:"):
            self.conversation_memory.add_message(role="user", content=query)
            self.conversation_memory.add_message(role="assistant", content=llm_response_or_stream)
        elif stream:
             self.conversation_memory.add_message(role="user", content=query)

        if stream:
            print("Warning: `process_query` called with `stream=True` but DeepEval evaluation expects final string. Ensure `evaluate_rag.py` collects the stream.")
            full_response_text = ""
            for chunk in llm_response_or_stream:
                full_response_text += chunk
            return full_response_text, context_texts_for_deepeval
        else:
            return llm_response_or_stream, context_texts_for_deepeval
            
    def add_assistant_response_to_memory(self, assistant_response: str):
        if assistant_response and not assistant_response.startswith("Error:"):
            self.conversation_memory.add_message(role="assistant", content=assistant_response)
        else:
            print(f"Skipping adding assistant response to memory due to empty or error response: {assistant_response}")

if __name__ == '__main__':
    print("RAG Pipeline Orchestrator Example")
    print("Ensure Ollama is running and vector store is built ('python build_vector_store.py').")

    import os
    if not os.path.exists("data/raw/orchestrator_example.txt"):
        os.makedirs("data/raw", exist_ok=True)
        with open("data/raw/orchestrator_example.txt", "w", encoding="utf-8") as f:
            f.write("This is a test document for the orchestrator. It discusses basic cybersecurity concepts like passwords and two-factor authentication (2FA).")
            f.write("\nAnother sentence about phishing awareness and how to spot fake emails.")
        
        if not os.path.exists("data/vector_store/faiss_index.index"):
            print("\nAttempting to build vector store for example...")
            try:
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from build_vector_store import main as build_main
                build_main()
                print("Vector store build script executed for example.")
            except ImportError:
                print("Could not import build_vector_store. Please run it manually.")
            except Exception as e:
                print(f"Error running build_vector_store for example: {e}")

    try:
        rag_system = RagPipeline(llm_model_name="qwen:0.5b") 

        print("\n--- Testing Non-Streaming Query ---")
        query1 = "What is two-factor authentication?"
        response1, context1 = rag_system.process_query(query1, stream=False, top_k_retrieval=3)
        print(f"\nQuery: {query1}")
        print(f"Response:\n{response1}")

        query2 = "Tell me about phishing."
        response2, context2 = rag_system.process_query(query2, stream=False, top_k_retrieval=3)
        print(f"\nQuery: {query2}")
        print(f"Response:\n{response2}")

        print("\n--- Testing Streaming Query ---")
        query3 = "What are common password security tips?"
        print(f"\nQuery: {query3}")
        print("Streamed Response:")
        
        stream_iterator_or_error, _ = rag_system.process_query(query3, stream=True, top_k_retrieval=3)
        
        full_streamed_response = ""
        if isinstance(stream_iterator_or_error, str): 
            print(stream_iterator_or_error)
            full_streamed_response = stream_iterator_or_error
        elif stream_iterator_or_error:
            for chunk in stream_iterator_or_error:
                print(chunk, end="", flush=True)
                full_streamed_response += chunk
            print("\n--- End of Stream ---")
            rag_system.add_assistant_response_to_memory(full_streamed_response)
        else:
            print("Stream iterator was None or empty.")

        print("\n--- Final Conversation History ---")
        for msg in rag_system.conversation_memory.get_history():
            print(f"  {msg['role'].capitalize()}: {msg['content'][:100]}...")

    except FileNotFoundError as e:
        print(f"\nError during RAG pipeline example: {e}")
        print("This often means the vector store is missing. Please run 'python build_vector_store.py' first after adding data to 'data/raw/'.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the RAG pipeline example: {e}")
        print("Check Ollama server status, model availability, and file paths.")