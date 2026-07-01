"""
Understanding Chains in LangChain V.1
LCEL patterns, composition, and debugging
"""

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)

load_dotenv()

# Initialize the model
model = init_chat_model(
    model="groq:llama-3.1-8b-instant",
    temperature=0,
)


def demo_basic_chain():
    """Basic LCEL Chain."""

    print("\n===== BASIC CHAIN =====")

    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence:\n\n{text}"
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke(
        {
            "text": "LangChain is a framework for developing applications powered by language models."
        }
    )

    print(f"\nSummary:\n{result}")


def demo_parallel_chain():
    """Run multiple chains in parallel."""

    print("\n===== PARALLEL CHAIN =====")

    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in two sentences:\n\n{text}"
    )

    keywords_prompt = ChatPromptTemplate.from_template(
        "Extract exactly 5 keywords from the following text.\n\n{text}\n\nReturn them as a comma-separated list."
    )

    sentiment_prompt = ChatPromptTemplate.from_template(
        "Determine the sentiment (Positive, Negative, or Neutral).\n\n{text}"
    )

    parser = StrOutputParser()

    analysis_chain = RunnableParallel(
        summary=summarize_prompt | model | parser,
        keywords=keywords_prompt | model | parser,
        sentiment=sentiment_prompt | model | parser,
    )

    text = """
    The new AI features are absolutely incredible! Users are loving the
    faster response times and improved accuracy. However, some have noted
    that the pricing could be more competitive. Overall, the product
    launch has been a massive success with record-breaking adoption rates.
    """

    results = analysis_chain.invoke({"text": text})

    print("\nAnalysis Results")
    print("-" * 50)
    print(f"Summary   : {results['summary']}")
    print(f"Keywords  : {results['keywords']}")
    print(f"Sentiment : {results['sentiment']}")


def demo_passthrough_chain():
    """Demonstrates RunnablePassthrough."""

    print("\n===== PASSTHROUGH CHAIN =====")

    prompt = ChatPromptTemplate.from_template(
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer the question using the context."
    )

    def fake_retriever(_):
        return "LangChain was created by Harrison Chase in 2022."

    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_retriever),
            question=RunnablePassthrough(),
        )
        | RunnableLambda(
            lambda x: {
                "context": x["context"],
                "question": x["question"]["question"],
            }
        )
        | prompt
        | model
        | StrOutputParser()
    )

    result = chain.invoke({"question": "Who created LangChain?"})

    print(f"\nAnswer:\n{result}")


def demo_chain_branching():
    """Demonstrates RunnableBranch."""

    print("\n===== CHAIN BRANCHING =====")

    code_prompt = ChatPromptTemplate.from_template(
        "You are a Python programming expert.\n\nQuestion: {input}"
    )

    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful AI assistant.\n\nQuestion: {input}"
    )

    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify the following question as either 'code' or 'general'.\n"
        "Return only one word.\n\n"
        "{input}"
    )

    classifier = classifier_prompt | model | StrOutputParser()

    def is_code_question(input_dict):
        classification = classifier.invoke(input_dict)
        return "code" in classification.lower()

    branch = RunnableBranch(
        (
            is_code_question,
            code_prompt | model | StrOutputParser(),
        ),
        general_prompt | model | StrOutputParser(),
    )

    questions = [
        "How do I write a for loop in Python?",
        "What's the weather like today?",
    ]

    for question in questions:
        print("\nQuestion:", question)
        answer = branch.invoke({"input": question})
        print("Answer:", answer)


def demo_debugging():
    """Demonstrates debugging techniques."""

    print("\n===== DEBUGGING CHAIN =====")

    prompt = ChatPromptTemplate.from_template(
        "Say hello to {name}"
    )

    chain = prompt | model | StrOutputParser()

    print("\nInput Schema")
    print("-" * 40)
    print(chain.input_schema.model_json_schema())

    print("\nOutput Schema")
    print("-" * 40)
    print(chain.output_schema.model_json_schema())

    result = chain.with_config(
        run_name="greeting_chain",
    ).invoke({"name": "Alice"})

    print("\nGreeting")
    print("-" * 40)
    print(result)

    def log_step(x, step_name=""):
        print(f"\n[{step_name}]")
        print(f"Type : {type(x).__name__}")
        print(f"Value: {str(x)[:150]}")
        return x

    debug_chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "After Prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "After Model"))
        | StrOutputParser()
    )

    print("\nExecuting Debug Chain...\n")

    result = debug_chain.invoke({"name": "Debug"})

    print("\nFinal Output")
    print("-" * 40)
    print(result)


def show_menu():
    """Display the menu."""

    print("\n" + "=" * 60)
    print("      Understanding Chains in LangChain (LCEL)")
    print("=" * 60)
    print("1. Basic Chain")
    print("2. Parallel Chain")
    print("3. RunnablePassthrough")
    print("4. RunnableBranch")
    print("5. Debugging a Chain")
    print("6. Run All Demos")
    print("0. Exit")
    print("=" * 60)


def main():
    demos = {
        "1": ("Basic Chain", demo_basic_chain),
        "2": ("Parallel Chain", demo_parallel_chain),
        "3": ("RunnablePassthrough", demo_passthrough_chain),
        "4": ("RunnableBranch", demo_chain_branching),
        "5": ("Debugging", demo_debugging),
    }

    while True:
        show_menu()

        choice = input("Enter your choice: ").strip()

        if choice == "0":
            print("\nThank you! Exiting...")
            break

        elif choice == "6":
            for _, (title, func) in demos.items():
                print("\n" + "#" * 70)
                print(f"Running: {title}")
                print("#" * 70)

                try:
                    func()
                except Exception as e:
                    print(f"\nError while running {title}: {e}")

                input("\nPress Enter to continue...")

        elif choice in demos:
            title, func = demos[choice]

            print("\n" + "#" * 70)
            print(f"Running: {title}")
            print("#" * 70)

            try:
                func()
            except Exception as e:
                print(f"\nError: {e}")

            input("\nPress Enter to continue...")

        else:
            print("\nInvalid choice! Please select a valid option.")


if __name__ == "__main__":
    main()