from typing import List, Dict, Union

Message = Dict[str, str]

class SimpleConversationMemory:
    def __init__(self, max_history_length: int = 10):
        self.history: List[Message] = []
        self.max_history_length = max_history_length

    def add_message(self, role: str, content: str):
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be 'user' or 'assistant'.")
        
        message: Message = {"role": role, "content": content}
        self.history.append(message)
        
        self._trim_history()

    def _trim_history(self):
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]

    def get_history(self) -> List[Message]:
        return self.history.copy()

    def get_formatted_history_for_prompt(self) -> str:
        formatted_string = ""
        for message in self.history:
            formatted_string += f"{message['role'].capitalize()}: {message['content']}\n"
        return formatted_string.strip()

    def clear_history(self):
        self.history = []
        print("Conversation history cleared.")

if __name__ == '__main__':
    memory = SimpleConversationMemory(max_history_length=4)

    memory.add_message("user", "Hello, who are you?")
    memory.add_message("assistant", "I am a helpful AI assistant.")
    
    print("History after Turn 1:")
    for msg in memory.get_history():
        print(msg)

    memory.add_message("user", "What can you do?")
    memory.add_message("assistant", "I can answer questions and provide information.")

    print("\nHistory after Turn 2 (should be full, 4 messages):")
    for msg in memory.get_history():
        print(msg)

    memory.add_message("user", "Tell me a joke.")
    memory.add_message("assistant", "Why don't scientists trust atoms? Because they make up everything!")

    print("\nHistory after Turn 3 (oldest turn should be gone):")
    for msg in memory.get_history():
        print(msg)
    
    assert len(memory.get_history()) <= memory.max_history_length

    memory.clear_history()
    print("\nHistory after clearing:")
    print(memory.get_history())
    assert len(memory.get_history()) == 0

    memory.add_message("user", "First message after clear.")
    print("\nHistory after adding one message post-clear:")
    for msg in memory.get_history():
        print(msg)
    assert len(memory.get_history()) == 1