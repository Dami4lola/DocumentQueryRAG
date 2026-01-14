import sys
import os

print("--- DIAGNOSTIC REPORT ---")
print(f"1. Python Executable: {sys.executable}")

try:
    import langchain
    print(f"2. LangChain Version: {langchain.__version__}")
    print(f"3. LangChain Location: {langchain.__file__}")
except ImportError:
    print("2. LangChain: NOT FOUND")

try:
    from langchain import chains
    print("4. langchain.chains: FOUND")
except ImportError as e:
    print(f"4. langchain.chains: FAILED ({e})")
    
print("-------------------------")