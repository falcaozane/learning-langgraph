from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import random
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ============================================================
# Select LLM Provider
# ============================================================


MODELS = {
    "groq": "groq:llama-3.1-8b-instant",
    "cerebras": "cerebras:zai-glm-4.7",
    "nvidia": "nvidia:deepseek-ai/deepseek-v4-flash",
}

# Randomly select a provider
provider = random.choice(list(MODELS.keys()))

print(f"Using provider: {provider}")
print(f"Model: {MODELS[provider]}")

# Initialize the model
model = init_chat_model(
    model=MODELS[provider],
    temperature=0,
)

# ============================================================
# ChatPromptTemplate Example
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "Translate the following text: {text}"),
    ]
)

messages = prompt.format_messages(
    input_language="English",
    output_language="French",
    text="I love programming.",
)

response = model.invoke(messages)

print("Translation:")
print(response.content)
print("-" * 60)

# ============================================================
# Few-shot Prompt Example
# ============================================================

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Give the opposite of each word."),
        fewshot_prompt,
        ("human", "{input}"),
    ]
)

response = model.invoke(
    final_prompt.format_messages(input="happy")
)

print("Few-shot Output:")
print(response.content)
print("-" * 60)

# ============================================================
# Reusable Prompt Components
# ============================================================

system_prompt = ChatPromptTemplate.from_messages(
    [("system", "You are a {role}.")]
)

user_prompt = ChatPromptTemplate.from_messages(
    [("human", "{question}")]
)

full_prompt = system_prompt + user_prompt

messages = full_prompt.format_messages(
    role="helpful assistant",
    question="What is AI?",
)

response = model.invoke(messages)

print("Reusable Prompt Output:")
print(response.content)