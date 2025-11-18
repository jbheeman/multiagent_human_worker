from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from knowledge_base.models import KnowledgeBase, ProductKnowledge, RefinementDecision, PurchasedProduct
from typing import List, Dict, Optional
import json
import re
from common_tools.logger import Logger

# This is the tool that the PersonaAgent will be forced to call in its 'refine' step.
@tool
def submit_refinement_decision(decision: RefinementDecision) -> str:
    """
    Submit the strategic decision for the next step of the research loop.

    Args:
        decision: A structured object containing the thought process, next research task,
                  keywords, pruning decisions, and the status of the refinement loop.
    """
    return "Refinement decision submitted successfully."


class PersonaAgent:
    """
    The 'brain' of the system, responsible for analyzing information and guiding the search
    based on a specific persona.
    """
    def __init__(self, model: OpenAIServerModel, user_products: Dict[str, PurchasedProduct], logger: Optional[Logger] = None):
        self.model = model
        self.logger = logger if logger else Logger()
        self.user_products = user_products
        self.personality_prompt = self.infer_persona_from_products(user_products)

        # This internal agent is specialized for making the strategic refinement decision.
        self._refinement_agent = ToolCallingAgent(
            tools=[submit_refinement_decision],
            model=model,
            max_steps=1, # Force it to call the tool immediately
        )

    def infer_persona_from_products(self, products: Dict[str, PurchasedProduct]) -> str:
        """
        Infers a user's persona from a list of their prior purchases.
        """
        product_list_str = "\n".join([f"- {p.title} (${p.price})" for p in products.values() if p.title])

        prompt = f"""
        Based on the following list of purchased products, please infer a persona for the user.

        **Purchased Products:**
        {product_list_str}

        **Instructions:**
        You must follow these steps and show your work for each one:
        1.  **Extract Traits:** For each product, identify key traits (e.g., brand, category, price point, features, implied hobbies or interests).
        2.  **Identify Buying Patterns:** Look for patterns across all products to determine the user's buying habits and preferences.
        3.  **Categorize Traits:** Group the user's inferred buying traits into the following four categories:
            - **Confident Likes:** Things we are confident the person likes.
            - **Somewhat Confident Likes:** Things we are somewhat confident the person likes.
            - **Confident Dislikes:** Things we are confident the person dislikes.
            - **Somewhat Confident Dislikes:** Things we are somewhat confident the person dislikes.
        4.  **Generate Persona Description:** Based on the categorized traits, write a concise, plaintext paragraph describing the user's persona. This description should be suitable for guiding a research assistant.

        **Output Format:**
        You must provide your full reasoning for steps 1-3. After your reasoning, provide the final persona description enclosed in `<persona_description>` tags.

        **Example Output:**
        **1. Extracted Traits:**
        - Airkeep Car Air Freshener: Low price, home/car accessory, scent-focused.
        - Lumiere & Co. Bike Seat Bag: Mid-range price, cycling accessory, practical.
        ...

        **2. Buying Patterns:**
        - The user frequently buys cycling-related gear, suggesting a hobby in cycling.
        - The user purchases items at various price points, but seems to value function over luxury.
        ...

        **3. Categorized Traits:**
        - **Confident Likes:** Cycling, practical items.
        - **Somewhat Confident Likes:** Home fragrance, pet safety.
        ...

        <persona_description>
        The user is a practical, budget-conscious individual who prioritizes functionality and value. They are an avid cyclist, investing in quality components for their hobby. They are not brand-loyal but seem to prefer items with good reviews and a focus on durability. They show some interest in home and pet accessories, but are not driven by luxury or high-end brands.
        </persona_description>
        """
        
        self.logger.log_overview("--- PersonaAgent: Inferring persona from prior purchases ---")
        
        # Log detailed prompt and output to a separate file
        inference_log_file = self.logger.get_log_file("PersonaInference")
        inference_log_file.write("--- Persona Inference Prompt ---\n")
        inference_log_file.write(prompt + "\n")
        inference_log_file.write("---------------------------------\n\n")
        
        response_message = self.model([{"role": "user", "content": prompt}])
        full_output = response_message.content
        
        inference_log_file.write("--- LLM Full Output ---\n")
        inference_log_file.write(full_output + "\n")
        inference_log_file.write("-----------------------\n")
        inference_log_file.close()

        match = re.search(r"<persona_description>(.*?)</persona_description>", full_output, re.DOTALL)
        if match:
            persona_description = match.group(1).strip()
            self.logger.log_overview(f"Inferred Persona: {persona_description}")
            return persona_description
        else:
            self.logger.log_overview("ERROR: Could not extract persona description from LLM output. Using full output as fallback.")
            # Fallback to using the full output if tags are not found
            return full_output

    def get_full_persona_prompt(self, task_prompt: str) -> str:
        """Helper to construct the full prompt including the persona."""
        product_titles = "\n".join([f"- {p.title}" for p in self.user_products.values() if p.title])
        
        return (
            f"**Persona Directive**\n"
            f"You are to embody the following persona. All of your analysis, decisions, and responses must be guided by this persona's traits and perspective.\n\n"
            f"**Persona Profile:**\n{self.personality_prompt}\n\n"
            f"---\n\n"
            f"**User's Prior Purchases:**\n{product_titles}\n\n"
            f"---\n\n"
            f"**Your Current Task:**\n{task_prompt}"
        )

    def initial_task(self, user_goal: str) -> str:
        """
        Formulates the best initial, broad research task based on the user goal and persona.
        """
        prompt = self.get_full_persona_prompt(
            f"""
            **Objective:** Formulate the best possible first step for a research assistant.

            **User Goal:** "{user_goal}"

            **Your Task:**
            Based on your persona and the user's goal, create a single, broad, initial research task for a research assistant.
            This first step should be about gathering a wide range of initial options, but it should be gently guided by your core values.
            Your response should be ONLY the text of the task itself.
            """
        )
        return self.model([{"role": "user", "content": prompt}])

    def identify_products_from_text(self, text: str) -> List[str]:
        """
        Extracts a list of unique product names from a block of text.
        """
        prompt = f"""
        From the following text, extract a list of unique product names (e.g., laptops, phones, etc.).
        Focus on specific model names and brands.
        Return ONLY a single JSON list of strings.

        Example Text: "The new MacBook Air M3 is great, but the Dell XPS 15 is also a contender. Some also consider the SuperBook Pro."
        Example Response:
        ```json
        ["MacBook Air M3", "Dell XPS 15", "SuperBook Pro"]
        ```

        Text to analyze:
        ---
        {text}
        ---
        """
        response_chat_message = self.model([{"role": "user", "content": prompt}])
        response_text = response_chat_message.content
        match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
        if not match:
            return []
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return []

    async def update_knowledge(self, product_name: str, existing_knowledge: ProductKnowledge, new_info: str) -> ProductKnowledge:
        """
        Parses new information and updates the pros, cons, and info for a single product.
        This includes adding new items, and removing or rephrasing existing ones.
        """
        import asyncio

        prompt = self.get_full_persona_prompt(
            f"""
            **Objective:** Update the knowledge for a single product based on new information.

            **Product Name:** {product_name}

            **Existing Knowledge:**
            - Pros: {existing_knowledge.pros}
            - Cons: {existing_knowledge.cons}
            - Info: {existing_knowledge.info}

            **New Information (raw text):**
            ---
            {new_info}
            ---

            **Your Task:**
            Critically analyze the 'New Information' and compare it to the 'Existing Knowledge'.
            Your goal is to produce the most accurate and concise knowledge base.

            1.  **Add:** If the new information provides a **new, meaningful pro, con, or factual detail** that is not already captured, add it to the appropriate list (`pros`, `cons`, `info`).
            2.  **Remove/Rephrase:** If the new information **invalidates, corrects, or provides a much better phrasing** for an existing item, you should remove the old item and add the new/rephrased one.
            3.  **Ignore:** If the new information is **redundant, irrelevant, or does not add value**, do nothing.

            Return ONLY a single JSON object in a markdown block. The JSON should contain:
            - `pros`, `cons`, `info`: Lists of new or rephrased items to **add**.
            - `remove_pros`, `remove_cons`, `remove_info`: Lists of existing items to **remove**.

            Example Response (Adding, removing, and rephrasing):
            ```json
            {{
                "pros": ["New, better-phrased pro"],
                "remove_pros": ["Old, poorly-phrased pro"],
                "cons": [],
                "remove_cons": [],
                "info": ["Updated factual detail"],
                "remove_info": ["Outdated factual detail"]
            }}
            ```

            Example Response (if no new info is found):
            ```json
            {{}}
            ```
            """
        )

        for attempt in range(2):
            # Use asyncio.to_thread to run the synchronous model call in a separate thread
            response_chat_message = await asyncio.to_thread(self.model, [{"role": "user", "content": prompt}])
            response_text = response_chat_message.content

            match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
            if match:
                try:
                    updated_data = json.loads(match.group(1))
                    
                    # Remove items first
                    for pro in updated_data.get("remove_pros", []):
                        if pro in existing_knowledge.pros:
                            existing_knowledge.pros.remove(pro)
                    for con in updated_data.get("remove_cons", []):
                        if con in existing_knowledge.cons:
                            existing_knowledge.cons.remove(con)
                    for item in updated_data.get("remove_info", []):
                        if item in existing_knowledge.info:
                            existing_knowledge.info.remove(item)

                    # Add new items and ensure no duplicates
                    existing_knowledge.pros = list(dict.fromkeys(existing_knowledge.pros + updated_data.get("pros", [])))
                    existing_knowledge.cons = list(dict.fromkeys(existing_knowledge.cons + updated_data.get("cons", [])))
                    existing_knowledge.info = list(dict.fromkeys(existing_knowledge.info + updated_data.get("info", [])))
                    
                    self.logger.log_overview(f"🧠 PersonaAgent: Successfully updated knowledge for {product_name}.")
                    return existing_knowledge # Success
                except json.JSONDecodeError:
                    error_feedback = f"Your previous response for {product_name} caused a JSON decoding error. Please ensure the JSON is perfectly formatted."
            else:
                error_feedback = f"Your previous response for {product_name} did not contain a valid JSON markdown block. Please provide one."

            # Prepare for retry
            prompt = f"RETRY ATTEMPT {attempt + 2}:\n{error_feedback}\n\nOriginal prompt:\n{prompt}"
            self.logger.log_overview(f"⚠️ PersonaAgent: update_knowledge attempt {attempt + 1} failed for {product_name}. Retrying...")

        self.logger.log_overview(f"⚠️ PersonaAgent: All update_knowledge attempts failed for {product_name}.")
        return existing_knowledge # Return original knowledge after all retries fail

    def refine(self, knowledge_base: KnowledgeBase, original_user_goal: str, last_decision: Optional[RefinementDecision] = None) -> RefinementDecision:
        """
        Analyzes the active items in the KnowledgeBase and determines the next strategic move.
        This is a global, strategic LLM call that is forced to return a structured
        RefinementDecision by using a tool-calling agent.
        """
        # Filter for active products only
        active_products = {name: product for name, product in knowledge_base.products.items() if product.status == 'researching'}
        
        kb_summary_parts = []
        for name, product in active_products.items():
            product_detail = f"  - **Product:** {name}\n"
            if product.pros:
                product_detail += "    - **Pros:**\n" + "\n".join(f"      - {p}" for p in product.pros) + "\n"
            if product.cons:
                product_detail += "    - **Cons:**\n" + "\n".join(f"      - {c}" for c in product.cons) + "\n"
            if product.info:
                product_detail += "    - **Info:**\n" + "\n".join(f"      - {i}" for i in product.info) + "\n"
            kb_summary_parts.append(product_detail)

        kb_summary = "\n".join(kb_summary_parts)
        if not kb_summary:
            kb_summary = "No active products are currently being researched."

        thought_history_summary = "\n".join(f"- {thought}" for thought in knowledge_base.thought_history)
        if not thought_history_summary:
            thought_history_summary = "No strategic decisions have been made yet."

        last_decision_summary = "No previous decision."
        if last_decision:
            last_decision_summary = f"""
            - **Previous Thought:** {last_decision.thought}
            - **Previous Task:** {last_decision.next_research_task}
            """

        prompt = self.get_full_persona_prompt(
            f"""
            **Objective:** Decide the next research step.

            **Original User Goal:** {original_user_goal}

            **Your Strategic History (Previous Thoughts & Lessons Learned):**
            {thought_history_summary}

            **Your Last Decision:**
            {last_decision_summary}

            **Current Viable Options:**
            {kb_summary}

            **Your Task:**
            Based on your strategic history, your last decision, and the current viable options, decide on the single most important next research task.
            - Do NOT repeat research paths from your strategic history or your last decision.
            - If options have incomplete information, your task should be to find it.
            - If the current options are not good enough, your task should be to find new ones.
            - If you have enough information, set the status to 'ready_to_choose'.
            - If any viable options are now unsuitable, add them to 'options_to_prune'.

            You MUST call the `submit_refinement_decision` tool to provide your answer. The `thought` field is critical as your "lesson learned" for this step.
            The `status` field must be either 'continue_refining' or 'ready_to_choose'.
            """
        )

        # Run the internal agent to get a structured decision, with retry logic
        for attempt in range(2): # Try up to 2 times
            self._refinement_agent.run(prompt)

            # Extract the structured data from the agent's tool call
            for step in self._refinement_agent.memory.steps:
                if hasattr(step, "tool_calls") and step.tool_calls:
                    for call in step.tool_calls:
                        if call.name == "submit_refinement_decision":
                            args = call.arguments
                            # Success, return the decision
                            return RefinementDecision(**args.get("decision", {}))
            
            # If we get here, the tool was not called. Prepare for retry.
            failed_output = self._refinement_agent.memory.steps[-1].model_output if self._refinement_agent.memory.steps else "No output."
            prompt = (
                f"Your previous response was invalid. You MUST call the `submit_refinement_decision` tool. Please try again.\n"
                f"Previous failed response: {failed_output}\n\n"
                f"--- Original Prompt ---\n{prompt}"
            )
            self.logger.log_overview(f"⚠️ PersonaAgent: refine attempt {attempt + 1} failed. Retrying...")
            self._refinement_agent.memory.reset() # Reset memory for a clean retry

        # Fallback if all attempts fail
        self.logger.log_overview("⚠️ PersonaAgent: All refine attempts failed to produce a tool call. Returning default stop decision.")
        return RefinementDecision(
            thought="Agent failed to generate a valid refinement decision after multiple attempts.",
            next_research_task="",
            status="ready_to_choose"
        )

    def choose(self, knowledge_base: KnowledgeBase) -> str:
        """
        Makes the final choice from the list of finalist products in the KnowledgeBase.
        """
        finalists = {name: product for name, product in knowledge_base.products.items() if product.status == 'researching'}

        if not finalists:
            return "No viable options were left to make a choice. The research may have been inconclusive or all options were pruned."

        finalist_summary = ""
        for name, product in finalists.items():
            pros_str = "\n".join(f"  - {pro}" for pro in product.pros)
            cons_str = "\n".join(f"  - {con}" for con in product.cons)
            finalist_summary += (
                f"**Product: {name}**\n"
                f"- **Pros:**\n{pros_str}\n"
                f"- **Cons:**\n{cons_str}\n\n"
            )

        prompt = self.get_full_persona_prompt(
            f"""
            **Objective:** Make a final decision.

            **Finalist Products:**
            ---
            {finalist_summary}
            ---

            **Your Task:**
            Based on your persona and the pros and cons of each finalist, choose the single best product.
            Your answer should be a single sentence stating your choice, followed by a brief justification.

            Example: "I choose the Framework Laptop 13 because its focus on repairability and recycled materials aligns perfectly with my values, despite the average battery life."
            """
        )

        # Use the base model for a direct chat completion call
        final_choice_message = self.model([{"role": "user", "content": prompt}])
        final_choice = final_choice_message.content
        self.logger.log_overview(f"🧠 PersonaAgent: Made final choice: {final_choice}")
        return final_choice

        def answer_question(self, question: str, context: str, knowledge_base: KnowledgeBase, last_decision: Optional[RefinementDecision]) -> str:

            """

            Answers a question from the manager agent's perspective, based on the persona.

            """

            kb_summary_parts = []

            for name, product in knowledge_base.products.items():

                product_detail = f"  - **Product:** {name}\n"

                if product.pros:

                    product_detail += "    - **Pros:**\n" + "\n".join(f"      - {p}" for p in product.pros) + "\n"

                if product.cons:

                    product_detail += "    - **Cons:**\n" + "\n".join(f"      - {c}" for c in product.cons) + "\n"

                if product.info:

                    product_detail += "    - **Info:**\n" + "\n".join(f"      - {i}" for i in product.info) + "\n"

                kb_summary_parts.append(product_detail)

    

            kb_summary = "\n".join(kb_summary_parts)

            if not kb_summary:

                kb_summary = "No products in the knowledge base yet."

    

            last_decision_summary = "No previous decision."

            if last_decision:

                last_decision_summary = f"""

                - **Previous Thought:** {last_decision.thought}

                - **Previous Task:** {last_decision.next_research_task}

                """

    

            prompt = self.get_full_persona_prompt(

                f"""

                **Objective:** Answer a clarifying question from your research assistant.

    

                **Full Knowledge Base:**

                {kb_summary}

    

                **Your Last Decision:**

                {last_decision_summary}

    

                **Assistant's Current Context:**

                ---

                {context}

                ---

    

                **Assistant's Question:** "{question}"

    

                **Your Task:**

                Based on your persona, the full knowledge base, your last decision, and the assistant's context and question, provide a clear and concise answer.

                """

            )

            self.logger.log_overview("--- Persona Agent: Answering Question ---")

            self.logger.log_overview(prompt)

            self.logger.log_overview("-----------------------------------------")

            response_message = self.model([{"role": "user", "content": prompt}])

            return response_message.content

    