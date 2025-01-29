from functools import cache
from typing import TypeAlias

from langchain_community.chat_models import FakeListChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from core.settings import settings
from schema.models import (
    AllModelEnum,
    FakeModelName,
    GroqModelName,
    OpenAIModelName,
)

# maps our model names to provider-specific model identifiers
_MODEL_TABLE = {
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",
    GroqModelName.LLAMA_31_8B: "llama-3.1-8b-instant",
    GroqModelName.LLAMA_33_70B: "llama-3.3-70b-versatile",
    GroqModelName.LLAMA_GUARD_3_8B: "llama-guard-3-8b",
    GroqModelName.LLAMA_DEEPSEEK_R1_70B: "deepseek-r1-distill-llama-70b",
    FakeModelName.FAKE: "fake",
}

ModelT: TypeAlias = ChatOpenAI | ChatGroq 
"""allowed model types returned by this factory"""


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    """cached factory providing configured model instances.
    
    returns ready-to-use chat model with provider-specific settings.
    cached to avoid redundant model initialization.
    """
    # note: models with streaming=true will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=true (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    # handle openai models with streaming enabled
    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
    
    # configure groq models with safety model exception
    if model_name in GroqModelName:
        if model_name == GroqModelName.LLAMA_GUARD_3_8B:
            return ChatGroq(model=api_model_name, temperature=0.0)  # safety model needs deterministic
        return ChatGroq(model=api_model_name, temperature=0.5)
    
    # simple fake model for testing
    if model_name in FakeModelName:
        return FakeListChatModel(responses=["This is a test response from the fake model."])