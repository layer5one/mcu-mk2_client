# /home/taylo/Cognition/cognition_client.py
import os
import json
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from llama_cpp import Llama

from orchestration import (
    TOOL_REGISTRY, 
    TASK_STATE, 
    dispatch_tool_call, 
    check_task_status
)
from tools.internal_tools import initialize_math_llm, calculate_internal
from memory.mega_memory_manager import MegaMemoryManager

# --- Configuration ---
PC_IP = "192.168.1.69"
PC_PORT = 8001
PRIMARY_LLM_PATH = "/home/taylo/Cognition/models/Qwen2-7B-Instruct-Q5_K_M.gguf"
MATH_LLM_PATH = "/home/taylo/Cognition/models/LFM2-350M-Math-Q4_K_M.gguf"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cognition_client")

# --- Globals ---
app = FastAPI()
PRIMARY_LLM = None
mega_memory = MegaMemoryManager()

# --- Models ---
class RegistrationRequest(BaseModel):
    device_id: str
    address: str
    tools: Dict

class MemorySyncRequest(BaseModel):
    memories: List

class AgentPromptRequest(BaseModel):
    prompt: str

# --- Internal Tool Definitions ---
INTERNAL_TOOLS = {
    "calculate_internal": calculate_internal,
    "check_task_status": check_task_status,
}

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    global PRIMARY_LLM
    logger.info("Cognition Client starting up...")
    try:
        # Load Primary LLM with significant offloading to GPU
        PRIMARY_LLM = Llama(
            model_path=PRIMARY_LLM_PATH,
            n_ctx=8192,
            n_gpu_layers=-1, # Offload all layers to GPU
            verbose=False
        )
        logger.info("Primary LLM loaded successfully.")
    except Exception as e:
        logger.fatal(f"FATAL: Could not load primary LLM: {e}")
    
    # Initialize co-processor models (lazy loaded on first use)
    initialize_math_llm(MATH_LLM_PATH)
    logger.info("Startup complete.")

# --- API Endpoints ---
@app.post("/register")
async def register_mcp_server(payload: RegistrationRequest):
    logger.info(f"Received registration from device: {payload.device_id} at {payload.address}")
    TOOL_REGISTRY[payload.device_id] = {
        "address": payload.address,
        "tools": payload.tools
    }
    logger.info(f"Current Tool Registry: {json.dumps(TOOL_REGISTRY, indent=2)}")
    return {"status": "success", "message": f"Device {payload.device_id} registered."}

@app.post("/memory/sync")
async def sync_memory(payload: MemorySyncRequest):
    logger.info(f"Received {len(payload.memories)} memories to sync.")
    success = mega_memory.add_memory_batch(payload.memories)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to sync memories.")
    return {"status": "success"}

@app.post("/agent/prompt")
async def run_agent_prompt(request: AgentPromptRequest):
    logger.info(f"Received new agent task: {request.prompt}")
    # Run the agent loop in the background to not block the HTTP response
    asyncio.create_task(agent_loop(request.prompt))
    return {"status": "success", "message": "Agent task started."}

# --- Agent Logic ---
def generate_topology_xml():
    xml = "<SystemTopology>\n"
    # Add Home PC tools
    xml += "  <Device>\n"
    xml += "    <ID>COGNITION_PC</ID>\n"
    xml += "    <Location>Local Machine, your core processing unit.</Location>\n"
    xml += "    <Tools>\n"
    xml += '      <Tool name="calculate_internal"><Description>An internal, high-speed tool for solving mathematical problems.</Description><Args>{"query": "string"}</Args></Tool>\n'
    xml += '      <Tool name="check_task_status"><Description>Checks the status of a background task. Use the task_id from a previous tool call.</Description><Args>{"task_id": "string"}</Args></Tool>\n'
    xml += "    </Tools>\n"
    xml += "  </Device>\n"

    # Add registered MCP server tools
    for device_id, info in TOOL_REGISTRY.items():
        xml += f"  <Device>\n"
        xml += f"    <ID>{device_id}</ID>\n"
        xml += f"    <Address>{info['address']}</Address>\n"
        xml += f"    <Tools>\n"
        for tool_name, tool_info in info['tools'].items():
            xml += f'      <Tool name="{tool_name}">\n'
            xml += f"        <Description>{tool_info['description']}</Description>\n"
            xml += f"        <Args>{json.dumps(tool_info['schema']['properties'])}</Args>\n"
            xml += f"      </Tool>\n"
        xml += f"    </Tools>\n"
        xml += f"  </Device>\n"
    xml += "</SystemTopology>"
    return xml

def construct_prompt(user_request, history):
    topology_xml = generate_topology_xml()
    system_prompt = """You are the Cognition Core, a master orchestrator for a distributed AI agent system. Your body is a network of devices, each with unique tools. Your mission is to solve user requests by creating a plan and executing it step-by-step using the available tools.

You MUST operate using the ReAct framework. For each step, you will:
1.  Output your reasoning inside a `<Thought>` block. Explain your plan and why you are choosing a specific tool on a specific device.
2.  Output a single, executable tool call inside an `<Action>` block. The action MUST be a valid JSON object with "device_id", "tool_name", and "args" keys. For internal tools on this PC, use device_id "COGNITION_PC".

After you act, the system will provide an `<Observation>` with the result. Continue this cycle until the user's request is fully resolved, then conclude your final thought with "I am done.".
"""
    
    prompt = f"""<|im_start|>system
{system_prompt}
{topology_xml}<|im_end|>
<|im_start|>user
{user_request}<|im_end|>
{history}"""
    return prompt

async def agent_loop(user_request: str):
    if not PRIMARY_LLM:
        logger.error("Agent loop cannot start: Primary LLM not loaded.")
        return

    history = ""
    max_turns = 10

    for i in range(max_turns):
        full_prompt = construct_prompt(user_request, history)
        
        logger.info(f"--- Agent Turn {i+1} ---")
        
        # Add the assistant prompt structure
        prompt_for_llm = full_prompt + "\n<|im_start|>assistant\n"

        output = PRIMARY_LLM(
            prompt_for_llm,
            max_tokens=1024,
            stop=["<|im_end|>", "<Observation>"],
            temperature=0.1,
            top_p=0.9
        )
        
        generated_text = output['choices']['text']
        history += "\n<|im_start|>assistant\n" + generated_text + "<|im_end|>"

        try:
            thought = generated_text.split("<Thought>").[1]split("</Thought>").strip()
            logger.info(f"THOUGHT: {thought}")
        except IndexError:
            logger.warning("Malformed response: No thought found. Ending loop.")
            break

        try:
            action_str = generated_text.split("<Action>").[1]split("</Action>").strip()
            logger.info(f"ACTION: {action_str}")
            action = json.loads(action_str)
            
            device_id = action.get("device_id")
            tool_name = action.get("tool_name")
            args = action.get("args", {})

            observation = ""
            if device_id == "COGNITION_PC":
                if tool_name in INTERNAL_TOOLS:
                    result = INTERNAL_TOOLS[tool_name](**args)
                    observation = json.dumps(result) if isinstance(result, dict) else str(result)
                else:
                    observation = f"Error: Internal tool '{tool_name}' not found."
            else:
                observation = await dispatch_tool_call(device_id, tool_name, args)

            logger.info(f"OBSERVATION: {observation}")
            history += f"\n<Observation>{observation}</Observation>"

            if "i am done" in thought.lower():
                logger.info("Agent concluded task.")
                break

        except (IndexError, json.JSONDecodeError) as e:
            logger.info(f"No valid action found or JSON parse error: {e}. Assuming final answer in thought.")
            break
            
    logger.info("--- Agent Loop Finished ---")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=PC_IP, port=PC_PORT)
