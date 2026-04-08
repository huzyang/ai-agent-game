from enum import Enum

class ExtendedModelType(Enum):
    GPT_4 = "gpt-4"
    QWEN3_5_FLASH = "qwen3.5-flash"
    QWEN3_5_MAX = "qwen3.5-max"

    @property
    def is_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model."""
        return self in {
            ExtendedModelType.GPT_4,
        }

    @property
    def token_limit(self) -> int:
        r"""Returns the maximum token limit for a given model.
        Returns:
            int: The maximum token limit for the given model.
        """
        if self is ExtendedModelType.GPT_4:
            return 8192
        elif self is ExtendedModelType.QWEN3_5_FLASH:
            return 4096
        else:
            raise ValueError("Unknown model type")
