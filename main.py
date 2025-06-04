import sys
from core_logic.orchestrator import RagPipeline

def run_cli():
    print("Initializing Cybersecurity RAG Bot...")
    print("This might take a moment as models and data are loaded...")
    
    try:
        rag_system = RagPipeline(
            llm_model_name="qwen:0.5b"
        )
    except FileNotFoundError as e:
        print(f"\nCritical Error: {e}")
        print("The vector store was not found. Please ensure you have run 'python build_vector_store.py'")
        print("after placing your .txt or .md documents in the 'data/raw/' directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\nCritical Error during RAG pipeline initialization: {e}")
        print("Please check your Ollama setup, model availability, and file paths.")
        sys.exit(1)

    print("\nCybersecurity RAG Bot is ready!")
    print("Type your questions below. Type 'exit' or 'quit' to end.")
    print("----------------------------------------------------")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting Cybersecurity RAG Bot. Goodbye!")
                break

            use_stream = True

            print("Bot: ", end="", flush=True)

            if use_stream:
                response_stream = rag_system.process_query(user_input, stream=True, top_k_retrieval=3)
                
                full_assistant_response = ""
                if isinstance(response_stream, str):
                    print(response_stream)
                elif response_stream:
                    try:
                        for chunk in response_stream:
                            print(chunk, end="", flush=True)
                            full_assistant_response += str(chunk)
                        print()
                        rag_system.add_assistant_response_to_memory(full_assistant_response)
                    except Exception as e:
                        print(f"\nError during response streaming: {e}")
                else:
                    print("Received no response stream.")
                
            else:
                response = rag_system.process_query(user_input, stream=False, top_k_retrieval=3)
                print(response)
            
            print("----------------------------------------------------")

        except KeyboardInterrupt:
            print("\nExiting due to KeyboardInterrupt. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")

if __name__ == "__main__":
    run_cli()