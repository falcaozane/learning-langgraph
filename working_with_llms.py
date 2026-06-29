"""
Working with LLMs in LangChain v1

Topics Covered
--------------
1. Initializing chat models
2. Comparing different providers
3. Working with message objects
4. Streaming responses
5. Multi-model exercise

Providers
---------
- NVIDIA
- Groq
- Cerebras
"""

from dotenv import load_dotenv

from langchain_nvidia import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_cerebras import ChatCerebras

from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


# ==========================================================
# Model Factory
# ==========================================================

def get_model(provider: str, temperature: float = 0.7, streaming=False):
    """Return a chat model based on provider."""

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
        raise ValueError("Unsupported provider")


# ==========================================================
# Demo 1
# ==========================================================

def demo_init_models():

    providers = [
        "nvidia",
        "groq",
        "cerebras",
    ]

    for provider in providers:

        print("=" * 60)
        print(provider.upper())

        model = get_model(provider, temperature=0)

        response = model.invoke(
            "What is the capital of France? Answer in one word."
        )

        print(response.content)


# ==========================================================
# Demo 2
# ==========================================================

def demo_model_comparison():

    prompt = "Explain recursion in one sentence."

    providers = [
        "nvidia",
        "groq",
        "cerebras",
    ]

    print(f"Prompt: {prompt}\n")

    for provider in providers:

        model = get_model(provider)

        response = model.invoke(prompt)

        print("=" * 60)
        print(provider.upper())
        print(response.content)
        print()


# ==========================================================
# Demo 3
# ==========================================================

def demo_messages(provider="groq"):

    model = get_model(provider, temperature=0)

    messages = [
        SystemMessage(
            content="You are a pirate. Always answer like a pirate."
        ),
        HumanMessage(
            content="What's the weather like today?"
        ),
    ]

    response = model.invoke(messages)

    print(response.content)

    messages.append(response)

    messages.append(
        HumanMessage(
            content="What about tomorrow?"
        )
    )

    print("\nConversation Continues\n")

    response = model.invoke(messages)

    print(response.content)


# ==========================================================
# Demo 4
# ==========================================================

def demo_streaming(provider="groq"):

    model = get_model(provider)

    print("=" * 60)
    print(f"Streaming using {provider.upper()}")
    print("=" * 60)

    for chunk in model.stream(
        "Write a short poem about AI."
    ):
        print(chunk.content, end="", flush=True)

    print()


# ==========================================================
# Exercise
# ==========================================================

def exercise_multi_model():

    providers = [
        "nvidia",
        "groq",
        "cerebras",
    ]

    question = "What is Artificial Intelligence?"

    responses = {}

    for provider in providers:

        model = get_model(provider)

        response = model.invoke(question)

        responses[provider] = response.content

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for provider, answer in responses.items():

        print(f"\n{provider.upper()}")
        print("-" * 30)
        print(answer)


# ==========================================================
# Connectivity Test
# ==========================================================

def test_models():

    providers = [
        "nvidia",
        "groq",
        "cerebras",
    ]

    for provider in providers:

        print("=" * 60)
        print(f"Testing {provider.upper()}")

        try:

            model = get_model(provider, temperature=0)

            response = model.invoke(
                "Say 'setup complete!' in one word."
            )

            print(response.content)

        except Exception as e:
            print(e)


# ==========================================================
# Menu
# ==========================================================

def choose_provider():

    print("\nChoose Provider")
    print("1. NVIDIA")
    print("2. Groq")
    print("3. Cerebras")

    choice = input("Choice: ")

    mapping = {
        "1": "nvidia",
        "2": "groq",
        "3": "cerebras",
    }

    return mapping.get(choice)


def main():

    while True:

        print("\n" + "=" * 60)
        print("Working with LLMs in LangChain")
        print("=" * 60)

        print("1. Test Providers")
        print("2. Initialize Models")
        print("3. Compare Models")
        print("4. Message Objects")
        print("5. Streaming")
        print("6. Exercise - Multi Model")
        print("0. Exit")

        choice = input("\nEnter choice: ")

        if choice == "0":
            break

        match choice:

            case "1":
                test_models()

            case "2":
                demo_init_models()

            case "3":
                demo_model_comparison()

            case "4":
                provider = choose_provider()
                if provider:
                    demo_messages(provider)

            case "5":
                provider = choose_provider()
                if provider:
                    demo_streaming(provider)

            case "6":
                exercise_multi_model()

            case _:
                print("Invalid choice.")


if __name__ == "__main__":
    main()