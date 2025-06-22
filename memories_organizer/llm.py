from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


@dataclass
class Description:
    description: str
    short_name: str


_DESCRIPTION_SCHEMA = [
    ResponseSchema(name="description", description="Detailed description of the file"),
    ResponseSchema(name="short_name", description="A short name describing the file"),
]

parser = StructuredOutputParser.from_response_schemas(_DESCRIPTION_SCHEMA)
FORMAT_INSTRUCTIONS = parser.get_format_instructions()


class LLMDescriber:
    def __init__(self, temperature: float = 0.2):
        self.llm = ChatOpenAI(temperature=temperature)

    def describe(self, info: str) -> Description:
        messages = [
            SystemMessage(content="You are a helpful assistant that summarizes media files."),
            HumanMessage(content=(
                f"{info}\n\n"
                f"{FORMAT_INSTRUCTIONS}"
            )),
        ]
        result = self.llm(messages)
        parsed = parser.parse(result.content)
        return Description(**parsed)
