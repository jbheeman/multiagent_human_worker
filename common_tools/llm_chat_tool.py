import os
from smolagents import OpenAIServerModel, tool
from typing import Optional
from planning_agent.plan_models import PlanState


# To handle the 'model' object, we use a factory function pattern.
def create_llm_chat_tool(model: OpenAIServerModel, plan_state=Optional[PlanState]):
    """
    Factory function that creates and returns the llm_chat tool.
    This pattern allows us to inject the model dependency into the tool's scope
    so it can be used within the decorated function.
    
    Args:
        model: The language model to use
        plan_state: Optional PlanState object to include results in context
    """
    
    @tool
    def llm_chat(query: str, context: str = "", include_prior_results: bool = False) -> str:
        """
        Use for general reasoning, summarization, or answering questions based on text.
        This tool is a direct interface to a powerful language model. It automatically
        receives the results from prior steps. It DOES NOT have access to the internet.
        
        Args:
            query (str): The specific question to ask or instruction to follow.
            context (str): Optional context (e.g., text from a file) to reason about.
            include_prior_results (bool): Populates the context with the results of prior steps if true.
        """
        # Build context from plan_state results if available
        full_context = context
        if plan_state and hasattr(plan_state, 'results') and plan_state.results and include_prior_results: 
            import json
            results_json = json.dumps(plan_state.results, indent=2)
            if full_context:
                full_context = f"Plan results:\n{results_json}\n\nAdditional context:\n{context}"
            else:
                full_context = f"Plan results:\n{results_json}"
        
        # Construct a clear prompt for the underlying model
        if full_context:
            prompt = f"Given the following context:\n\n---\n{full_context}\n---\n\nPlease respond to the following query: {query}"
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

