import ollama
from typing import List, Dict, Any, Optional, Union, Iterator

class LLMInterface:
    def __init__(self, model_name: str, ollama_host: Optional[str] = None):
        self.model_name = model_name
        self.client = ollama.Client(host=ollama_host if ollama_host else 'http://localhost:11434')
        self._check_model_availability()

    def _check_model_availability(self):
        try:
            models_info = self.client.list()
            print(f"DEBUG: Full models_info from client.list(): {models_info}")

            ollama_models_list = models_info.get('models', [])
            print(f"DEBUG: Parsed ollama_models_list: {ollama_models_list}")

            available_models = []
            for model_entry in ollama_models_list:
                model_name_str = None
                if isinstance(model_entry, dict): # Standard check for ollama library
                    model_name_str = model_entry.get('name')
                    if not isinstance(model_name_str, str): # Ensure the value is a string
                        model_name_str = None
                
                if not model_name_str and hasattr(model_entry, 'name'): # Check for .name attribute
                    if isinstance(model_entry.name, str):
                        model_name_str = model_entry.name
                
                if not model_name_str and hasattr(model_entry, 'model'): # Fallback to .model attribute
                    if isinstance(model_entry.model, str):
                         model_name_str = model_entry.model

                if model_name_str:
                    available_models.append(model_name_str)
                else:
                    print(f"Warning: Unexpected model entry format or unable to extract name: {model_entry}")

            base_model_name = self.model_name.split(':')[0]
            is_available = any(
                self.model_name == am or
                (base_model_name == am.split(':')[0] if ':' in am else False)
                for am in available_models
            )

            if not is_available:
                print(f"INFO: Model '{self.model_name}' not found in locally parsed list. Attempting to use with Ollama server.")
            else:
                print(f"INFO: LLMInterface initialized for model: {self.model_name}. Model found in locally parsed list.")

        except ollama.ResponseError as e:
            print(f"ERROR: Could not connect to Ollama or list models: {e}")
            if hasattr(e, 'status_code') and e.status_code == 404 and "model" in str(e).lower() and "not found" in str(e).lower():
                print(f"ERROR: Server indicated model '{self.model_name}' was not found.")
            print("Ensure the Ollama server is running and accessible.")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while checking model availability: {e} (Type: {type(e).__name__})")
            
    def generate_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None
    ) -> Union[str, Iterator[str]]:
        if not prompt:
            return "Error: Prompt cannot be empty."

        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({'role': 'user', 'content': prompt})

        try:
            print(f"\nSending prompt to LLM ({self.model_name}):")

            if stream:
                response_stream = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    options=options
                )
                def content_stream_generator():
                    for chunk in response_stream:
                        if not chunk.get('done', False) and 'content' in chunk.get('message', {}):
                            yield chunk['message']['content']
                        elif chunk.get('done', False):
                            break
                return content_stream_generator()
            else:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    options=options
                )
                return response['message']['content']

        except ollama.ResponseError as e:
            error_message_detail = e.error if hasattr(e, 'error') else str(e)
            status_code = e.status_code if hasattr(e, 'status_code') else 'N/A'
            error_message = f"Ollama API Response Error: {status_code} - {error_message_detail}"
            print(error_message)
            if "model not found" in error_message_detail.lower():
                 error_message += f". Please ensure model '{self.model_name}' is pulled and Ollama is running."
            return error_message
        except Exception as e:
            error_message = f"An unexpected error occurred during LLM interaction: {e}"
            print(error_message)
            return error_message

if __name__ == '__main__':
    print("LLMInterface Example:")
    llm_model_to_test = "qwen:0.5b"

    try:
        llm_interface = LLMInterface(model_name=llm_model_to_test)
        
        print("\n--- Test 1: Simple Prompt (No Stream) ---")
        simple_prompt = "What is the capital of France?"
        response1 = llm_interface.generate_response(simple_prompt)
        print(f"LLM Response to '{simple_prompt}':\n{response1}")

        print("\n--- Test 2: Prompt with History (No Stream) ---")
        history = [
            {'role': 'user', 'content': 'What are the three main types of cyber attacks?'},
            {'role': 'assistant', 'content': 'The three main types are often categorized as attacks on confidentiality, integrity, and availability.'}
        ]
        follow_up_prompt = "Can you give an example for an attack on confidentiality?"
        response2 = llm_interface.generate_response(follow_up_prompt, conversation_history=history)
        print(f"LLM Response to '{follow_up_prompt}' (with history):\n{response2}")

        print("\n--- Test 3: Simple Prompt (With Stream) ---")
        streaming_prompt = "Explain the concept of a firewall in simple terms."
        print(f"LLM Streaming Response to '{streaming_prompt}':")
        response_stream_gen = llm_interface.generate_response(streaming_prompt, stream=True)
        
        if isinstance(response_stream_gen, str):
            print(response_stream_gen)
        else:
            for chunk_content in response_stream_gen:
                print(chunk_content, end="", flush=True)
            print("\n--- End of Stream ---")

        print("\n--- Test 4: Prompt with Options (No Stream) ---")
        creative_prompt = "Write a very short story about a mischievous AI."
        llm_options = {"temperature": 0.9, "num_predict": 100}
        response4 = llm_interface.generate_response(creative_prompt, options=llm_options)
        print(f"LLM Creative Response (temp {llm_options.get('temperature', 'default')}):\n{response4}")

    except Exception as e:
        print(f"\nAn error occurred during the LLMInterface example: {e}")
        print("Please ensure Ollama is running, the model is pulled, and network is accessible if needed.")