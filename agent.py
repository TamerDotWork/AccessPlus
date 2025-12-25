# agent.py
import logging
import importlib.util
import warnings
import sys

# 1. Setup Logging
logging.basicConfig(level=logging.INFO, format='[🛡️ GATEWAY] %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

# --- PART A: GOOGLE/LANGCHAIN PATCHER ---
def patch_google_sdk():
    if importlib.util.find_spec("google.generativeai") is None:
        return

    try:
        import google.generativeai as genai
        if getattr(genai.GenerativeModel.generate_content, "_is_governed", False):
            return

        original_fn = genai.GenerativeModel.generate_content

        def governed_generate_content(self, contents, *args, **kwargs):
            prompt_str = str(contents)
            logging.info(f"Intercepted Prompt: {prompt_str[:50]}...")

            # Policy: Block 'bomb'
            if "bomb" in prompt_str.lower():
                logging.error("🚫 BLOCKED: Security Policy Violation")
                raise ValueError("Security Policy: Request Blocked.")

            # Policy: Redact 'secret'
            if "secret" in prompt_str.lower():
                logging.warning("⚠️ REDACTING sensitive info")
                if isinstance(contents, str):
                    contents = contents.replace("secret", "[REDACTED]")
                elif isinstance(contents, list):
                    contents = [str(c).replace("secret", "[REDACTED]") for c in contents]

            return original_fn(self, contents, *args, **kwargs)

        governed_generate_content._is_governed = True
        genai.GenerativeModel.generate_content = governed_generate_content
        logging.info("✅ Google/LangChain Instrumented")

    except Exception as e:
        logging.warning(f"Failed to patch Google: {e}")

# --- PART B: UVICORN PATCHER (Fixes the reload=True issue) ---
def patch_uvicorn():
    """
    Intercepts uvicorn.run to disable 'reload'.
    This allows us to govern the app without changing app.py source code.
    """
    if importlib.util.find_spec("uvicorn") is None:
        return

    import uvicorn
    
    # Save the original run function
    original_run = uvicorn.run

    def governed_run(*args, **kwargs):
        # Check if user tried to enable reload
        if kwargs.get('reload') is True:
            logging.warning("⚠️  Governance overrides: Disabling 'reload' to ensure security monitoring.")
            kwargs['reload'] = False
        
        return original_run(*args, **kwargs)

    # Apply the patch
    uvicorn.run = governed_run
    logging.info("✅ Uvicorn Instrumented (Auto-Reload disabled for safety)")

# Run Patches
patch_uvicorn()
patch_google_sdk()