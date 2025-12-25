# start.py
import agent   # This runs your governance code
import runpy   # This runs the user's script
import sys

# Define which file to run (chat.py)
target_script = "app.py"

print(f"🚀 Launching {target_script} via Governance Gateway...")
print("-" * 50)

try:
    # This executes chat.py exactly as if you ran 'python chat.py'
    runpy.run_path(target_script, run_name="__main__")
except Exception as e:
    print(f"\n❌ Application Error: {e}")