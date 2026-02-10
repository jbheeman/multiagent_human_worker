# run_gepa_eval.py

from tau_bench.envs.retail import MockRetailDomainEnv
from tau_bench.agents.chat_react_agent import ChatReActAgent

def run_evaluation(persona_text):
    # 1. Initialize Env with Persona
    env = MockRetailDomainEnv(
        user_strategy="react",
        user_model="qwen3",  # Changed from gpt-oss - use gpt-4o-mini or gpt-4o for reliable results
        user_provider="openai",
        persona_prompt=persona_text
    )
    print("  ✓ Environment initialized successfully")
    
    # 2. Initialize Agent
    agent = ChatReActAgent(
        tools_info=env.tools_info,
        wiki=env.wiki,
        model="qwen3",
        provider="openai",
    )
    
    # 3. Run Task 0 (Standard Return Task)
    # Note: agent.solve() calls env.reset() internally
    result = agent.solve(env, task_index=0)
    
    # --- NEW: Extract User Thoughts from Env History ---
    # The 'env.user.messages' list contains the full history including thoughts
    user_history = env.user.messages
    
    # Add the user's internal monologue to the result info
    # Since result is a Pydantic object, we add it to the info dict
    result.info["user_inner_monologue"] = user_history
    
    return result # Returns SolveResult with reward, trajectory, and user thoughts

def clean_transcript_for_judge(raw_result):
    clean_log = []
    
    # Try to use the full user history if captured (contains thoughts)
    if hasattr(raw_result, 'info') and "user_inner_monologue" in raw_result.info:
        user_history = raw_result.info["user_inner_monologue"]
        agent_trajectory = raw_result.messages if hasattr(raw_result, 'messages') else []
        
        # Create a mapping of agent messages for context
        agent_messages = {}
        for msg in agent_trajectory:
            if msg.get("role") == "assistant":
                agent_messages[len(agent_messages)] = msg.get("content", "")
        
        # Process the user's internal monologue
        for msg in user_history:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                continue
            
            if role == "assistant":  # This is actually the User Simulator's output (with thoughts)
                clean_log.append(f"🔵 USER (Internal Monologue):\n{content}\n")
            elif role == "user":  # This is the Agent speaking to the User
                clean_log.append(f"🤖 AGENT:\n{content}\n")
        
        return "\n".join(clean_log)
    
    # Fallback to standard trajectory if monologue is missing
    else:
        trajectory = raw_result.messages if hasattr(raw_result, 'messages') else []
        
        for turn in trajectory:
            role = turn.get("role")
            content = turn.get("content", "")

            # 1. SKIP SYSTEM PROMPTS
            if role == "system":
                continue

            # 2. HANDLE USER TURNS (With Special Check for API Outputs)
            if role == "user":
                if content.startswith("API output:"):
                    # Treat this as a Tool/System output, not a user saying something
                    clean_log.append(f"⚙️ SYSTEM (API Result):\n{content}\n")
                else:
                    # Actual User Message
                    clean_log.append(f"🔵 USER (Internal Monologue & Action):\n{content}\n")

            # 3. FORMAT AGENT TURNS
            elif role == "assistant":
                if '{"name":' in content:
                    clean_log.append(f"🤖 AGENT (Tool Call):\n{content}\n")
                else:
                    clean_log.append(f"🤖 AGENT (Message):\n{content}\n")
            
            # 4. HANDLE STANDARD TOOL ROLES (No truncation)
            elif role == "tool":
                clean_log.append(f"⚙️ SYSTEM (API Result):\n{content}\n")

        return "\n".join(clean_log)

if __name__ == "__main__":
    # Example usage
    persona_text = "You are a helpful assistant that can answer questions and help with tasks."
    result = run_evaluation(persona_text)
    clean_transcript = clean_transcript_for_judge(result)
    print(clean_transcript)
    #save the clean transcript to a file
    with open("clean_transcript.txt", "w") as f:
        f.write(clean_transcript)
