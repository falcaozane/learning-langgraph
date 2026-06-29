from dotenv import load_dotenv

load_dotenv()

from langchain_core import __version__ as core_version
#from langgraph import __version__ as lg_version
from importlib.metadata import version
lg_version = version("langgraph")

from langchain_nvidia import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_cerebras import ChatCerebras


print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")



def main():

    # Testing nvidia
    llm_nvidia = ChatNVIDIA(model_name="deepseek-ai/deepseek-v4-flash", temperature=0)
    response = llm_nvidia.invoke("Say 'setup complete!' in one word")
    print(f"Response from NVIDIA: {response}")

    print("=*50")

    # Testing cerebras
    llm_cerebras = ChatCerebras(model="zai-glm-4.7", temperature=0)
    response = llm_cerebras.invoke("Say 'setup complete!' in one word")
    print(f"Response from Cerebras: {response}")

    print("=*50")

    # Testing groq
    llm_groq = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    response = llm_groq.invoke("Say 'setup complete!' in one word")
    print(f"Response from GROQ: {response}")

    print("=*50")




if __name__ == "__main__":
    main()
