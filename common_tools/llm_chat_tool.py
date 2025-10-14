import os
from smolagents import OpenAIServerModel, tool


# To handle the 'model' object, we use a factory function pattern.
def create_llm_chat_tool(model: OpenAIServerModel):
    """
    Factory function that creates and returns the llm_chat tool.
    This pattern allows us to inject the model dependency into the tool's scope
    so it can be used within the decorated function.
    """
    
    @tool
    def llm_chat(query: str, context: str = "") -> str:
        """
        Use for general reasoning, summarization, or answering questions based on text.
        This tool is a direct interface to a powerful language model.
        
        Args:
            query (str): The specific question to ask or instruction to follow.
            context (str): Optional context (e.g., text from a file) to reason about.
        """
        # Construct a clear prompt for the underlying model
        if context:
            prompt = f"Given the following context:\n\n---\n{context}\n---\n\nPlease respond to the following query: {query}"
        else:
            prompt = query

        messages = [{"role": "user", "content": prompt}]
        response = model(messages)
        
        return response.content

    return llm_chat

# This block allows the tool to be tested in isolation.
if __name__ == "__main__":
    """
    A simple, self-contained test for the LLMChatTool.
    """
    print("--- Testing LLM Chat Tool ---")
    
    try:
        chat_model = OpenAIServerModel(
            model_id="gemma3",
            api_base="https://ellm.nrp-nautilus.io/v1",
            api_key=os.environ.get("NAUT_API_KEY", "your_default_api_key"),
        )

        test_tool = create_llm_chat_tool(model=chat_model)

        test_query = "What is the primary function of a mitochondria in a cell?"
        print(f"\nQuery: {test_query}")

        # 4. Call the tool directly
        # The @tool decorator makes the returned function object callable.
        result = test_tool(query=test_query)

        # 5. Print the result
        print(f"\nResult: {result}")
        print("\n--- Test Complete ---")

    except Exception as e:
        print(f"\n--- Test Failed ---")
        print(f"An error occurred: {e}")

