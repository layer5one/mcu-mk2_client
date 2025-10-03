# /home/taylo/Cognition/tools/internal_tools.py
from llama_cpp import Llama
import logging

logger = logging.getLogger("cognition_client")

MATH_COPROCESSOR_LLM = None

def initialize_math_llm(model_path: str):
    global MATH_COPROCESSOR_LLM
    if MATH_COPROCESSOR_LLM is None:
        try:
            MATH_COPROCESSOR_LLM = Llama(
                model_path=model_path,
                n_ctx=1024,
                verbose=False
            )
            logger.info("Math co-processor model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load math co-processor model: {e}")

def calculate_internal(query: str) -> str:
    """
    An internal, high-speed tool for solving mathematical problems.
    Use this for any numerical calculations.
    """
    if MATH_COPROCESSOR_LLM is None:
        return "Error: Math co-processor is not available."

    prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        output = MATH_COPROCESSOR_LLM(
            prompt,
            max_tokens=256,
            temperature=0.1,
            stop=["<|im_end|>"]
        )
        return output['choices']['text'].strip()
    except Exception as e:
        return f"Error during internal calculation: {e}"
