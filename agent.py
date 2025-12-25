# agent.py
import logging
import importlib.util
import warnings
import sys

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, format='[🛡️ GATEWAY] %(message)s')

# 2. Fix the Google Warning
#    This line hides the "All support for google.generativeai has ended" message
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

def patch_google_sdk():
    """Patches Google GenAI (works for both Old and New SDKs)"""
    
    # Check if the library is installed
    if importlib.util.find_spec("google.generativeai") is None:
        return

    try:
        import google.generativeai as genai
        
        # Avoid double-patching
        if getattr(genai.GenerativeModel.generate_content, "_is_governed", False):
            return

        original_fn = genai.GenerativeModel.generate_content

        def governed_generate_content(self, contents, *args, **kwargs):
            # --- GOVERNANCE ---
            logging.info(f"Intercepted Prompt: {str(contents)[:50]}...")
            
            # Policy Example: Block the word "secret"
            if "secret" in str(contents).lower():
                logging.error("BLOCKED: Policy Violation")
                raise ValueError("Sensitive data detected in prompt.")

            # --- EXECUTE ---
            response = original_fn(self, contents, *args, **kwargs)

            # --- MONITOR ---
            logging.info("Response received successfully.")
            return response

        # Apply Patch
        governed_generate_content._is_governed = True
        genai.GenerativeModel.generate_content = governed_generate_content
        logging.info("✅ Agent Attached: Governance Active")
        
    except Exception as e:
        logging.warning(f"Agent failed to attach: {e}")

# Run the patcher immediately when this file is imported
patch_google_sdk()