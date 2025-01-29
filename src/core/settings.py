from typing import Annotated, Any

from dotenv import find_dotenv
from pydantic import BeforeValidator, HttpUrl, SecretStr, TypeAdapter, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from schema.models import (
    AllModelEnum,
    FakeModelName,
    GroqModelName,
    OpenAIModelName,
    Provider,
)


# validate that a string is a valid http url
def check_str_is_http(x: str) -> str:
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


# settings class to manage environment configuration
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(),  # locate the .env file
        env_file_encoding="utf-8",  # set encoding for the .env file
        env_ignore_empty=True,  # ignore empty environment variables
        extra="ignore",  # ignore extra fields not defined in the class
        validate_default=False,  # disable default validation
    )
    MODE: str | None = None  # mode of the application (e.g., dev, prod)

    HOST: str = "0.0.0.0"  # default host
    PORT: int = 80  # default port

    AUTH_SECRET: SecretStr | None = None  # secret for authentication

    OPENAI_API_KEY: SecretStr | None = None  # openai api key
    GROQ_API_KEY: SecretStr | None = None  # groq api key
    USE_FAKE_MODEL: bool = False  # flag to use a fake model

    # default model to use (can be set in the post-initialization method)
    DEFAULT_MODEL: AllModelEnum | None = None  # type: ignore[assignment]
    AVAILABLE_MODELS: set[AllModelEnum] = set()  # type: ignore[assignment]

    OPENWEATHERMAP_API_KEY: SecretStr | None = None  # openweathermap api key

    LANGCHAIN_TRACING_V2: bool = False  # flag for langchain tracing
    LANGCHAIN_PROJECT: str = "default"  # default langchain project
    LANGCHAIN_ENDPOINT: Annotated[str, BeforeValidator(check_str_is_http)] = (
        "https://api.smith.langchain.com"
    )  # langchain api endpoint
    LANGCHAIN_API_KEY: SecretStr | None = None  # langchain api key

    # post-initialization method for settings
    def model_post_init(self, __context: Any) -> None:
        # dictionary to hold provider keys
        api_keys = {
            Provider.OPENAI: self.OPENAI_API_KEY,
            Provider.GROQ: self.GROQ_API_KEY,
            Provider.FAKE: self.USE_FAKE_MODEL,
        }
        # collect active keys
        active_keys = [k for k, v in api_keys.items() if v]
        if not active_keys:
            raise ValueError("At least one LLM API key must be provided.")  # ensure at least one key is provided

        # iterate over active providers and configure settings
        for provider in active_keys:
            match provider:
                case Provider.OPENAI:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = OpenAIModelName.GPT_4O_MINI  # default model for openai
                    self.AVAILABLE_MODELS.update(set(OpenAIModelName))  # add available openai models               
                case Provider.GROQ:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = GroqModelName.LLAMA_31_8B  # default model for groq
                    self.AVAILABLE_MODELS.update(set(GroqModelName))  # add available groq models
                case Provider.FAKE:
                    if self.DEFAULT_MODEL is None:
                        self.DEFAULT_MODEL = FakeModelName.FAKE  # default fake model
                    self.AVAILABLE_MODELS.update(set(FakeModelName))  # add available fake models
                case _:
                    raise ValueError(f"Unknown provider: {provider}")  # handle unknown provider

    # computed property to generate base url
    @computed_field
    @property
    def BASE_URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"  # format the base url

    # check if the application is in development mode
    def is_dev(self) -> bool:
        return self.MODE == "dev"


# create an instance of the settings
settings = Settings()
