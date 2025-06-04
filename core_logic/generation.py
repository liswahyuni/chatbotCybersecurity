import os
from typing import List, Tuple, Dict, Optional, Union, Iterator, Any
from rag_pipeline.llm_interface import LLMInterface
from rag_pipeline.conversation_memory import Message
from rag_pipeline.vector_store_manager import LangchainDocument

class ResponseGenerator:
    def __init__(self, llm_interface: LLMInterface):
        if not isinstance(llm_interface, LLMInterface):
            raise TypeError("llm_interface must be an instance of LLMInterface")
        self.llm_interface = llm_interface

    def _format_retrieved_documents_for_prompt(self, retrieved_docs: List[Tuple[LangchainDocument, float]]) -> str:
        if not retrieved_docs:
            return "No relevant documents found."

        context_str = "--- Relevant Context Extracted From Documents ---\n"
        for i, (doc, score) in enumerate(retrieved_docs):
            context_str += f"\n[Context Snippet {i+1} from: {doc['metadata'].get('source', 'Unknown Source')}]\n"
            context_str += f"{doc['page_content']}\n"
            context_str += "---------------------------------------------\n"
        return context_str

    def construct_prompt_with_context(
        self,
        user_query: str,
        retrieved_docs_with_scores: List[Tuple[LangchainDocument, float]],
        conversation_history: List[Message]
    ) -> List[Message]:
        system_prompt_content = (
            "You are a highly specialized AI assistant focused on providing precise, factual, and concise answers "
            "to technical questions specifically about Cybersecurity, Pentesting, and Hacking. "
            "Use ONLY the provided context documents to formulate your answer. "
            "The user is asking a specific technical question. Your task is to answer this question directly and accurately "
            "based SOLELY on the information presented in the 'CONTEXT DOCUMENTS' section. "
            "Do NOT add any conversational fluff, apologies, or summaries of the context itself. "
            "Focus entirely on extracting or inferring the direct answer to the 'USER QUERY' from the 'CONTEXT DOCUMENTS'. "
            "If the 'CONTEXT DOCUMENTS' do not contain the information to answer the 'USER QUERY', "
            "respond with ONLY the phrase: 'Information not available in the provided context.' "
            "Do not attempt to answer from general knowledge. Do not make up information."
        )
        messages: List[Message] = [{"role": "system", "content": system_prompt_content}]

        if conversation_history:
            messages.extend(conversation_history)

        formatted_context = self._format_retrieved_documents_for_prompt(retrieved_docs_with_scores)

        final_user_content = (
            f"CONTEXT DOCUMENTS:\n"
            f"---------------------\n"
            f"{formatted_context}"
            f"---------------------\n\n"
            f"USER QUERY: \"{user_query}\"\n\n"
            f"Based strictly on the CONTEXT DOCUMENTS provided above, what is the direct answer to the USER QUERY? "
            f"Your Answer:"
        )
        messages.append({"role": "user", "content": final_user_content})

        return messages

    def get_augmented_response(
        self,
        user_query: str,
        retrieved_docs_with_scores: List[Tuple[LangchainDocument, float]],
        conversation_history: List[Message],
        stream: bool = False,
        llm_options: Optional[Dict[str, Any]] = None
    ) -> Union[str, Iterator[str]]:
        if not user_query:
            return "Error: User query cannot be empty."

        messages_for_llm = self.construct_prompt_with_context(
            user_query,
            retrieved_docs_with_scores,
            conversation_history
        )
        
        history_up_to_now = [msg for msg in messages_for_llm if msg['role'] != 'user' or msg['content'] != messages_for_llm[-1]['content']]
        current_prompt_with_context = messages_for_llm[-1]['content']
        
        actual_history_for_llm = messages_for_llm[:-1]

        response = self.llm_interface.generate_response(
            prompt=current_prompt_with_context,
            conversation_history=actual_history_for_llm,
            stream=stream,
            options=llm_options
        )
        return response

if __name__ == '__main__':
    print("ResponseGenerator Example:")
    
    class MockLLMInterface:
        def __init__(self, model_name="mock_model"):
            self.model_name = model_name
            print(f"MockLLMInterface initialized with {model_name}")

        def generate_response(self, prompt: str, conversation_history: Optional[List[Message]] = None, stream: bool = False, options: Optional[Dict[str, Any]]=None):
            print("\n--- MockLLMInterface.generate_response called ---")
            print(f"Conversation History Passed to LLM ({len(conversation_history) if conversation_history else 0} messages):")
            if conversation_history:
                for msg in conversation_history:
                    print(f"  Role: {msg['role']}, Content Snippet: {msg['content'][:70]}...")
            print(f"Prompt Passed to LLM (User Query + Context):\n{prompt[:300]}...")
            print("-------------------------------------------------")
            if stream:
                def stream_gen():
                    yield "This is a "
                    yield "mocked streamed "
                    yield "response."
                return stream_gen()
            return "This is a mocked LLM response to the augmented prompt."

    llm_interface_instance = MockLLMInterface()
    response_gen = ResponseGenerator(llm_interface=llm_interface_instance)

    sample_user_query = "How to prevent XSS attacks?"
    sample_retrieved_docs: List[Tuple[LangchainDocument, float]] = [
        (
            {"page_content": "Cross-Site Scripting (XSS) can be mitigated by validating user input, encoding output, and using Content Security Policy (CSP).", 
             "metadata": {"source": "owasp_xss.txt"}},
            0.25
        ),
        (
            {"page_content": "Sanitizing inputs is crucial for preventing XSS. Always treat user-supplied data as untrusted.", 
             "metadata": {"source": "security_best_practices.md"}},
            0.30
        )
    ]
    sample_conversation_history: List[Message] = [
        {"role": "user", "content": "What are common web vulnerabilities?"},
        {"role": "assistant", "content": "Common web vulnerabilities include XSS, SQL Injection, and CSRF."}
    ]

    print("\n--- Testing Prompt Construction ---")
    constructed_messages = response_gen.construct_prompt_with_context(
        sample_user_query,
        sample_retrieved_docs,
        sample_conversation_history
    )
    print("Constructed Messages for LLM:")
    for i, msg in enumerate(constructed_messages):
        print(f"Message {i}: Role: {msg['role']}")
        print(f"  Content: {msg['content'][:200]}...")
    
    print("\n--- Testing Augmented Response (No Stream) ---")
    final_response = response_gen.get_augmented_response(
        sample_user_query,
        sample_retrieved_docs,
        sample_conversation_history,
        stream=False
    )
    print(f"\nFinal Mocked Response:\n{final_response}")

    print("\n--- Testing Augmented Response (Stream) ---")
    streamed_response_gen = response_gen.get_augmented_response(
        sample_user_query,
        sample_retrieved_docs,
        sample_conversation_history,
        stream=True
    )
    print("\nStreamed Mocked Response:")
    if isinstance(streamed_response_gen, str):
        print(streamed_response_gen)
    else:
        for chunk in streamed_response_gen:
            print(chunk, end="", flush=True)
        print("\n--- End of Stream ---")