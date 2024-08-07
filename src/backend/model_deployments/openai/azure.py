import json
import os
from typing import (
    Any,
    AsyncGenerator,
    Coroutine,
    Dict,
    List,
    Literal,
    Sequence,
    TypedDict,
)
import uuid

from openai import (
    NOT_GIVEN,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AsyncStream,
)
from openai.types.chat import (
    chat_completion as sync_chat,
)
from openai.types.chat import chat_completion_chunk as stream_chat
from openai.types.chat import chat_completion_message_tool_call_param as chat_tool_call
from cohere.core import request_options as cohere_request_options
from cohere import types as cohere_types

from backend.chat.collate import to_dict
from backend.model_deployments.openai.tools import TOOL_CONFIGS, ToolName
from backend.schemas import chat as chat_schemas
from backend.model_deployments.base import BaseDeployment
from backend.model_deployments.utils import get_model_config_var
from backend.schemas.cohere_chat import CohereChatRequest
from backend.services.logger import get_logger, send_log_message

MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION = "2024-02-15-preview"
USER_AGENT = "Chatbot/AsyncAzureOpenAI/1.0.0"

AZURE_OPENAI_API_VERSION_ENV_VAR = "AZURE_OPENAI_API_VERSION"
AZURE_OPENAI_API_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"
AZURE_OPENAI_RESOURCE_ENV_VAR = "AZURE_OPENAI_RESOURCE"
AZURE_OPENAI_MODEL_ENV_VAR = "AZURE_OPENAI_MODEL"

AZURE_OPENAI_ENV_VARS = [
    AZURE_OPENAI_API_VERSION_ENV_VAR,
    AZURE_OPENAI_API_KEY_ENV_VAR,
    AZURE_OPENAI_RESOURCE_ENV_VAR,
    AZURE_OPENAI_MODEL_ENV_VAR,
]

logger = get_logger()


class OpenAIMessage(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_call_id: str | None
    name: str | None


class ToolCallData(TypedDict):
    name: str
    parameters: dict[str, Any]


class ToolResultData(TypedDict):
    call: ToolCallData
    outputs: Sequence[dict[str, Any]]


def _init_openai_client(**kwargs: Any) -> tuple[AsyncOpenAI, str]:
    # API version check
    api_version = get_model_config_var(AZURE_OPENAI_API_VERSION_ENV_VAR, **kwargs)
    if not api_version:
        raise Exception(f"{AZURE_OPENAI_API_VERSION_ENV_VAR} is required")
    if api_version < MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION:
        raise Exception(
            f"The minimum supported Azure OpenAI preview API version is '{MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION}'"
        )

    # Endpoint
    openai_resource = get_model_config_var(AZURE_OPENAI_RESOURCE_ENV_VAR, **kwargs)
    if not openai_resource:
        raise Exception(f"{AZURE_OPENAI_RESOURCE_ENV_VAR} is required")
    openai_endpoint = f"https://{openai_resource}.openai.azure.com/"

    # Authentication
    openai_key = get_model_config_var(AZURE_OPENAI_API_KEY_ENV_VAR, **kwargs)
    if not openai_key:
        raise Exception(f"{AZURE_OPENAI_API_KEY_ENV_VAR} is required")

    # Deployment
    deployment = get_model_config_var(AZURE_OPENAI_MODEL_ENV_VAR, **kwargs)
    if not deployment:
        raise Exception(f"{AZURE_OPENAI_MODEL_ENV_VAR} is required")

    # Default Headers
    default_headers = {"x-ms-useragent": USER_AGENT}

    azure_openai_client = AsyncAzureOpenAI(
        api_version=api_version,
        api_key=openai_key,
        default_headers=default_headers,
        azure_endpoint=openai_endpoint,
    )
    return azure_openai_client, deployment


def _build_parameter(
    message: str,
    preamble: str | None,
    chat_history: list[OpenAIMessage],
    tools: Sequence[cohere_types.Tool],
    tool_results: Sequence[ToolResultData],
) -> tuple[dict[str, Any], Sequence[OpenAIMessage]]:
    messages: list[OpenAIMessage] = chat_history[:]
    if message:
        messages.append(
            {
                "role": "user",
                "content": message,
            }
        )
    if preamble:
        messages.insert(
            0,
            {
                "role": "system",
                "content": preamble,
            },
        )

    for tool_result in tool_results:
        tool_call: dict = tool_result.get("call", {})
        tool_call_meta: dict = tool_call.get("parameters", {}).get(
            "_tool_call_meta", {}
        )
        messages.append(
            {
                "role": "tool",
                "content": json.dumps(tool_result.get("outputs")),
                "tool_call_id": tool_call_meta.get("tool_call_id"),
                "name": tool_call.get("name"),
            }
        )
    if not tools or not chat_history or tool_results:
        return {}, messages

    tools_system_messages = [msg for msg in chat_history if msg["role"] == "system"]
    tools_system_messages[:0] = [
        {
            "role": "system",
            "content": msg,
        }
        for msg in [
            "Don't make assumptions about what values to plug into functions.",
            "Please determine the values to plug into functions.",
        ]
    ]

    tools_config = [
        TOOL_CONFIGS[tool.name] for tool in tools if tool.name in TOOL_CONFIGS
    ]

    return {
        "tools": tools_config,
        "tool_choice": "auto",
    }, (tools_system_messages if tools_config else messages)


def _to_openai_message(
    message: chat_schemas.ChatMessage | OpenAIMessage,
) -> OpenAIMessage:
    message_params = {}
    if isinstance(message, chat_schemas.ChatMessage):
        content = message.message
        role = message.role
    else:
        content = message.get("message")
        role = message.get("role")
        # only step 2+ will have tool_calls
        tool_calls = []
        for tool_call in message.get("tool_calls") or []:
            tool_call_meta = tool_call.get("parameters", {}).get("_tool_call_meta")
            tool_calls.append(
                chat_tool_call.ChatCompletionMessageToolCallParam(
                    id=tool_call_meta.get("tool_call_id"),
                    type="function",
                    function=chat_tool_call.Function(
                        arguments=tool_call_meta.get("function"),
                        name=tool_call_meta.get("name"),
                    ),
                )
            )
        if tool_calls:
            message_params["tool_calls"] = tool_calls
    match role:
        case "USER":
            role = "user"
        case "SYSTEM":
            role = "system"
        case "TOOL":
            role = "tool"
        case "CHATBOT":
            role = "assistant"
    return {"role": role, "content": content, **message_params}


class OpenAIAzureDeployment(BaseDeployment):
    def __init__(self, **kwargs: Any):
        self.client, self.deployment = _init_openai_client(**kwargs)
        conversation_id = kwargs.get("conversation_id")
        send_log_message(
            logger, "OpenAIAzureDeployment initialized", conversation_id=conversation_id
        )

    def _parse_result(self, completion: sync_chat.ChatCompletion):
        choice = completion.choices[0] if completion.choices else None

        if not choice:
            yield cohere_types.StreamedChatResponse_StreamStart(
                generation_id=str(uuid.uuid4())
            )
            return

        if isinstance(choice, sync_chat.Choice):
            tool_call = next(iter(choice.message.tool_calls or []), None)
            if tool_call:
                yield cohere_types.StreamedChatResponse_ToolCallsGeneration(
                    text=choice.message.content,
                    tool_calls=[
                        cohere_types.ToolCall(
                            name=tool_call.function.name,
                            parameters=json.loads(tool_call.function.arguments),
                        )
                    ],
                )

        finish_reason: cohere_types.ChatStreamEndEventFinishReason | None = None
        match choice.finish_reason:
            case "stop":
                finish_reason = "COMPLETE"
            case "length":
                finish_reason = "MAX_TOKENS"
            case "tool_calls":
                yield cohere_types.StreamedChatResponse_ToolCallsGeneration(
                    tool_calls=[],
                )
            case "function_call" | "content_filter" | _:
                pass
        if finish_reason:
            yield cohere_types.StreamedChatResponse_StreamEnd(
                finish_reason=finish_reason,
                response=cohere_types.NonStreamedChatResponse(
                    text=choice.message.content or "",
                    generation_id=str(uuid.uuid4()),
                    response_id=str(uuid.uuid4()),
                    finish_reason=finish_reason,
                ),
            )

    def _parse_stream_result(
        self,
        completion: stream_chat.ChatCompletionChunk,
        text_chunks: list[cohere_types.StreamedChatResponse_TextGeneration],
        tool_chunks: list[cohere_types.StreamedChatResponse_ToolCallsChunk],
        chat_history: Sequence[cohere_types.Message],
        tools: dict[str, cohere_types.Tool],
    ):
        choice = completion.choices[0] if completion.choices else None

        if not choice:
            yield cohere_types.StreamedChatResponse_StreamStart(
                generation_id=str(uuid.uuid4())
            )
            return

        tool_call = next(iter(choice.delta.tool_calls or []), None)
        if tool_call and tool_call.function:
            yield cohere_types.StreamedChatResponse_ToolCallsChunk(
                tool_call_delta=cohere_types.ToolCallDelta(
                    name=tool_call.function.name,
                    index=tool_call.index,
                    parameters=tool_call.function.arguments,
                    text=tool_call.id,
                ),
            )

        content = choice.delta.content or ""
        if content:
            yield cohere_types.StreamedChatResponse_TextGeneration(
                text=content or "",
            )

        finish_reason: cohere_types.ChatStreamEndEventFinishReason | None = None
        tool_calls: list[cohere_types.ToolCall] | None = None
        match choice.finish_reason:
            case "stop":
                message = "".join((t.text for t in text_chunks))
                chat_history = [
                    *chat_history,
                    cohere_types.Message_Chatbot(message=message),
                ]
                finish_reason = "COMPLETE"
            case "length":
                finish_reason = "MAX_TOKENS"
            case "tool_calls":
                tool_call_name = ""
                tool_parameters_string = ""
                tool_call_id = ""
                for t in tool_chunks:
                    tool_call_id = t.tool_call_delta.text or tool_call_id
                    tool_call_name = t.tool_call_delta.name or tool_call_name
                    tool_parameters_string += t.tool_call_delta.parameters or ""
                tool = tools.get(tool_call_name)
                content = content or (tool.description if tool else "")
                tool_params = {
                    **json.loads(tool_parameters_string),
                    "_tool_call_meta": {
                        "tool_call_id": tool_call_id,
                        "function": tool_parameters_string,
                        "name": tool_call_name,
                    },
                }
                tool_calls = [
                    cohere_types.ToolCall(
                        name=tool_call_name,
                        parameters=tool_params,
                    )
                ]
                chat_history = [
                    *chat_history,
                    cohere_types.Message_Chatbot(message="", tool_calls=tool_calls),
                ]
                yield cohere_types.StreamedChatResponse_ToolCallsGeneration(
                    text=content, tool_calls=tool_calls
                )
                finish_reason = "COMPLETE"
            case "function_call" | "content_filter" | _:
                pass
        if finish_reason:
            yield cohere_types.StreamedChatResponse_StreamEnd(
                finish_reason=finish_reason,
                response=cohere_types.NonStreamedChatResponse(
                    text=choice.delta.content or "",
                    generation_id=str(uuid.uuid4()),
                    response_id=str(uuid.uuid4()),
                    finish_reason=finish_reason,
                    chat_history=chat_history,
                    tool_calls=tool_calls,
                    meta=(
                        cohere_types.ApiMeta(
                            tokens=cohere_types.ApiMetaTokens(
                                input_tokens=completion.usage.prompt_tokens,
                                output_tokens=completion.usage.completion_tokens,
                            ),
                        )
                        if completion.usage
                        else None
                    ),
                ),
            )

    async def _process_stream_response(
        self, chat_request: CohereChatRequest, response: AsyncStream
    ):
        text_chunks: list[cohere_types.StreamedChatResponse_TextGeneration] = []
        tool_chunks: list[cohere_types.StreamedChatResponse_ToolCallsChunk] = []
        tools = {tool.name: tool for tool in chat_request.tools or []}
        async for completionChunk in response:
            for stream_event in self._parse_stream_result(
                completionChunk,
                tool_chunks=tool_chunks,
                text_chunks=text_chunks,
                chat_history=chat_request.chat_history,
                tools=tools,
            ):
                # stream only events
                if isinstance(
                    stream_event, cohere_types.StreamedChatResponse_ToolCallsChunk
                ):
                    tool_chunks.append(stream_event)
                elif isinstance(
                    stream_event, cohere_types.StreamedChatResponse_TextGeneration
                ):
                    text_chunks.append(stream_event)
                yield to_dict(stream_event)

    def _chat(
        self,
        *,
        message: str,
        chat_history: Sequence[OpenAIMessage],
        stream: bool = False,
        model: str | None = None,
        preamble: str | None = None,
        prompt_truncation: cohere_types.ChatStreamRequestPromptTruncation | None = None,
        connectors: Sequence[cohere_types.ChatConnector] | None = None,
        search_queries_only: bool | None = None,
        documents: Sequence[cohere_types.ChatDocument] | None = None,
        citation_quality: cohere_types.ChatStreamRequestCitationQuality | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_input_tokens: int | None = None,
        k: int | None = None,
        p: float | None = None,
        seed: int | None = None,
        stop_sequences: Sequence[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        raw_prompting: bool | None = None,
        return_prompt: bool | None = None,
        tools: Sequence[cohere_types.Tool] | None = None,
        tool_results: Sequence[cohere_types.ToolResult] | None = None,
        force_single_step: bool | None = None,
        request_options: cohere_request_options.RequestOptions | None = None,
    ) -> Coroutine[Any, Any, AsyncStream] | sync_chat.ChatCompletion:
        tools_params, messages = _build_parameter(
            message=message,
            preamble=preamble,
            chat_history=chat_history,
            tools=tools or [],
            tool_results=tool_results or [],
        )
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=NOT_GIVEN if temperature is None else float(temperature),
            top_p=NOT_GIVEN if p is None else float(p),
            stream=stream,
            **tools_params,
        )

    async def _chat_stream(
        self,
        *,
        message: str,
        stream: bool = False,
        model: str | None = None,
        preamble: str | None = None,
        chat_history: Sequence[OpenAIMessage] | None = None,
        prompt_truncation: cohere_types.ChatStreamRequestPromptTruncation | None = None,
        connectors: Sequence[cohere_types.ChatConnector] | None = None,
        search_queries_only: bool | None = None,
        documents: Sequence[cohere_types.ChatDocument] | None = None,
        citation_quality: cohere_types.ChatStreamRequestCitationQuality | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_input_tokens: int | None = None,
        k: int | None = None,
        p: float | None = None,
        seed: int | None = None,
        stop_sequences: Sequence[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        raw_prompting: bool | None = None,
        return_prompt: bool | None = None,
        tools: Sequence[cohere_types.Tool] | None = None,
        tool_results: Sequence[cohere_types.ToolResult] | None = None,
        force_single_step: bool | None = None,
        request_options: cohere_request_options.RequestOptions | None = None,
    ) -> AsyncStream:
        return await self._chat(
            message=message,
            stream=stream,
            model=model,
            preamble=preamble,
            chat_history=chat_history,
            prompt_truncation=prompt_truncation,
            connectors=connectors,
            search_queries_only=search_queries_only,
            documents=documents,
            citation_quality=citation_quality,
            temperature=temperature,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            k=k,
            p=p,
            seed=seed,
            stop_sequences=stop_sequences,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            raw_prompting=raw_prompting,
            return_prompt=return_prompt,
            tools=tools,
            tool_results=tool_results,
            force_single_step=force_single_step,
            request_options=request_options,
        )

    @property
    def rerank_enabled(self) -> bool:
        return False

    @classmethod
    def list_models(cls) -> List[str]:
        if not cls.is_available():
            return []

        return ["chatgpt-35"]

    @classmethod
    def is_available(cls) -> bool:
        return all([os.environ.get(var) is not None for var in AZURE_OPENAI_ENV_VARS])

    async def invoke_chat(self, chat_request: CohereChatRequest) -> Any:
        response = self.client.chat(
            **chat_request.model_dump(exclude={"stream", "file_ids", "agent_id"}),
        )
        yield to_dict(response)

    async def invoke_chat_stream(
        self, chat_request: CohereChatRequest, **kwargs: Any
    ) -> AsyncGenerator[Any, Any]:
        try:
            response = await self._chat_stream(
                model=self.deployment,
                stream=True,
                chat_history=[_to_openai_message(m) for m in chat_request.chat_history],
                tools=chat_request.tools,
                **chat_request.model_dump(
                    exclude={
                        "chat_history",
                        "tools",
                        "stream",
                        "file_ids",
                        "conversation_id",
                        "agent_id",
                        "model",
                    }
                ),
            )
            async for stream_event in self._process_stream_response(
                chat_request, response
            ):
                yield stream_event
        except Exception as e:
            logger.exception(e)
            raise e

    async def invoke_rerank(self, query: str, documents: List[Dict[str, Any]]) -> Any:
        return None
