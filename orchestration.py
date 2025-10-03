# /home/taylo/Cognition/orchestration.py
import httpx
import uuid
import asyncio
import logging
from typing import Dict

logger = logging.getLogger("cognition_client")

# --- Global State ---
TOOL_REGISTRY: Dict = {}
TASK_STATE: Dict = {}

async def dispatch_tool_call(device_id: str, tool_name: str, params: dict) -> str:
    task_id = str(uuid.uuid4())
    
    if device_id not in TOOL_REGISTRY:
        error_msg = f"Error: Device '{device_id}' not registered."
        logger.error(error_msg)
        return error_msg
        
    device_info = TOOL_REGISTRY[device_id]
    tool_url = f"{device_info['address']}/tools/{tool_name}"
    
    TASK_STATE[task_id] = {
        "status": "PENDING",
        "device_id": device_id,
        "tool_name": tool_name,
        "params": params,
        "result": None
    }
    
    asyncio.create_task(execute_http_call(task_id, tool_url, params))
    
    logger.info(f"Task {task_id} dispatched to {device_id} to run {tool_name}.")
    # The agent will need to check the status of this task later.
    return f"Task {task_id} is now running. Use the 'check_task_status' tool to get the result."

async def execute_http_call(task_id: str, url: str, params: dict):
    TASK_STATE[task_id]["status"] = "RUNNING"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json={"params": params})
            response.raise_for_status()
            TASK_STATE[task_id]["status"] = "SUCCESS"
            TASK_STATE[task_id]["result"] = response.json()
    except Exception as e:
        TASK_STATE[task_id]["status"] = "FAILURE"
        TASK_STATE[task_id]["result"] = str(e)
        logger.error(f"HTTP call for task {task_id} failed: {e}")
    
    logger.info(f"Task {task_id} finished with status: {TASK_STATE[task_id]['status']}")

def check_task_status(task_id: str) -> dict:
    """Checks the status and result of a previously dispatched asynchronous task."""
    return TASK_STATE.get(task_id, {"status": "NOT_FOUND"})
