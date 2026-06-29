"""
LangChain LCEL Demo
Supports multiple providers:
- NVIDIA
- Groq
- Cerebras

Required Environment Variables

NVIDIA_API_KEY=...
GROQ_API_KEY=...
CEREBRAS_API_KEY=...
"""

from dotenv import load_dotenv

from langchain_nvidia import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_cerebras import ChatCerebras

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ==========================================================
# Universal Model Loader
# ==========================================================

def get_llm(provider: str = "groq", temperature: float = 0.7):
    """Return an initialized chat model."""

    provider = provider.lower()

    if provider == "nvidia":
        return ChatNVIDIA(
            model_name="deepseek-ai/deepseek-v4-flash",
            temperature=temperature,
        )

    elif provider == "groq":
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=temperature,
        )

    elif provider == "cerebras":
        return ChatCerebras(
            model="zai-glm-4.7",
            temperature=temperature,
        )

    else:
        raise ValueError(
            "Provider must be one of: nvidia, groq, cerebras"
        )


# ==========================================================
# Demo 1 - Basic Chain
# ==========================================================

def demo_basic_chain(provider="groq"):
    """Demonstrates a basic LCEL chain."""

    print(f"\n{'='*60}")
    print(f"Basic Chain ({provider.upper()})")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence.\n\nQuestion: {question}"
    )

    model = get_llm(provider)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke(
        {
            "question": "What is LangChain?"
        }
    )

    print(result)

    return chain


# ==========================================================
# Demo 2 - Batch Execution
# ==========================================================

def demo_batch_execution(provider="groq"):
    """Demonstrate batch execution."""

    print(f"\n{'='*60}")
    print(f"Batch Execution ({provider.upper()})")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template(
        "Translate the following text to French:\n{text}"
    )

    chain = (
        prompt
        | get_llm(provider)
        | StrOutputParser()
    )

    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]

    results = chain.batch(inputs)

    for inp, out in zip(inputs, results):
        print(f"\nInput : {inp['text']}")
        print(f"Output: {out}")


# ==========================================================
# Demo 3 - Streaming
# ==========================================================

def demo_streaming(provider="groq"):
    """Demonstrate streaming output."""

    print(f"\n{'='*60}")
    print(f"Streaming ({provider.upper()})")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template(
        "Write a haiku about {topic}"
    )

    chain = (
        prompt
        | get_llm(provider)
        | StrOutputParser()
    )

    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)

    print("\n")


# ==========================================================
# Demo 4 - Schema Inspection
# ==========================================================

def demo_schema_inspection(provider="groq"):
    """Display chain input/output schema."""

    print(f"\n{'='*60}")
    print(f"Schema Inspection ({provider.upper()})")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template(
        "Summarize:\n{text}"
    )

    chain = (
        prompt
        | get_llm(provider)
        | StrOutputParser()
    )

    print("\nInput Schema")
    print("-" * 40)
    print(chain.input_schema.model_json_schema())

    print("\nOutput Schema")
    print("-" * 40)
    print(chain.output_schema.model_json_schema())


# ==========================================================
# Exercise
# ==========================================================

def exercise_first_chain(provider="groq"):
    """
    Create a chain that:
    1. Takes a product name
    2. Takes a target audience
    3. Generates a marketing tagline
    """

    print(f"\n{'='*60}")
    print(f"Exercise ({provider.upper()})")
    print("=" * 60)

    prompt = ChatPromptTemplate.from_template(
        """
Create a short, catchy marketing tagline.

Product:
{product}

Audience:
{audience}

Return ONLY the tagline.
"""
    )

    chain = (
        prompt
        | get_llm(provider)
        | StrOutputParser()
    )

    result = chain.invoke(
        {
            "product": "AI Course",
            "audience": "developers",
        }
    )

    print(result)


# ==========================================================
# Test Each Provider
# ==========================================================

def test_models():
    """Quick connectivity test for each provider."""

    print("\nTesting available providers...\n")

    providers = [
        "nvidia",
        "cerebras",
        "groq",
    ]

    for provider in providers:

        print("=" * 60)
        print(f"Testing {provider.upper()}")

        try:
            llm = get_llm(provider, temperature=0)

            response = llm.invoke(
                "Say 'setup complete!' in one word."
            )

            print("Response:")
            print(response.content)

        except Exception as e:
            print(f"Error: {e}")


def main():
    while True:
        print("\n" + "=" * 60)
        print("LangChain LCEL Demo")
        print("=" * 60)
        print("1. Test Model Connectivity")
        print("2. Basic Chain")
        print("3. Batch Execution")
        print("4. Streaming")
        print("5. Schema Inspection")
        print("6. Marketing Tagline Exercise")
        print("7. Run All Demos")
        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("Goodbye!")
            break

        provider = None

        # Don't ask provider when testing all models
        if choice != "1":
            print("\nSelect Provider")
            print("1. NVIDIA")
            print("2. Groq")
            print("3. Cerebras")

            provider_choice = input("Enter provider: ").strip()

            provider_map = {
                "1": "nvidia",
                "2": "groq",
                "3": "cerebras",
            }

            provider = provider_map.get(provider_choice)

            if provider is None:
                print("Invalid provider!")
                continue

        match choice:

            case "1":
                test_models()

            case "2":
                demo_basic_chain(provider)

            case "3":
                demo_batch_execution(provider)

            case "4":
                demo_streaming(provider)

            case "5":
                demo_schema_inspection(provider)

            case "6":
                exercise_first_chain(provider)

            case "7":
                demo_basic_chain(provider)
                demo_batch_execution(provider)
                demo_streaming(provider)
                demo_schema_inspection(provider)
                exercise_first_chain(provider)

            case _:
                print("Invalid choice!")


if __name__ == "__main__":
    main()
