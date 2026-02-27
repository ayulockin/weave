import os
from collections.abc import Generator
from importlib.metadata import version
from typing import Any

import litellm
import pytest
from litellm.utils import Delta, ModelResponse, StreamingChoices, Usage
from packaging.version import parse as version_parse

import weave
from weave.integrations.litellm.litellm import get_litellm_patcher, litellm_accumulator
from weave.integrations.openai.openai_sdk import get_openai_patcher

# This PR:
# https://github.com/BerriAI/litellm/commit/fe2aa706e8ff4edbcd109897e5da6b83ef6ad693
# Changed the output format for OpenAI to use APIResponse.
# We can handle this in non-streaming mode, but in streaming mode, we
# have no way of correctly capturing the output and not messing up the
# users' code (that i can see). In these cases, model cost is not captured.
USES_RAW_OPENAI_RESPONSE = version_parse(version("litellm")) > version_parse("1.42.11")


class Nearly:
    def __init__(self, v: float) -> None:
        self.v = v

    def __eq__(self, other: Any) -> bool:
        return abs(self.v - other) < 2


@pytest.fixture(scope="package")
def patch_litellm(request: Any) -> Generator[None, None, None]:
    # This little hack is to allow us to run the tests in prod mode
    # For some reason pytest's import procedure causes the patching
    # to fail in prod mode. Specifically, the patches get run twice
    # despite the fact that the patcher is a singleton.
    weave_server_flag = request.config.getoption("--trace-server")
    if weave_server_flag == ("prod"):
        yield
        return

    # Patch both LiteLLM and OpenAI since LiteLLM uses OpenAI as backend
    litellm_patcher = get_litellm_patcher()
    openai_patcher = get_openai_patcher()

    litellm_patcher.attempt_patch()
    openai_patcher.attempt_patch()

    yield

    litellm_patcher.undo_patch()
    openai_patcher.undo_patch()


@pytest.mark.skip_clickhouse_client  # TODO:VCR recording does not seem to allow us to make requests to the clickhouse db in non-recording mode
@pytest.mark.vcr(
    filter_headers=["authorization"], allowed_hosts=["api.wandb.ai", "localhost"]
)
def test_litellm_quickstart(
    client: weave.trace.weave_client.WeaveClient, patch_litellm: None
) -> None:
    # This is taken directly from https://docs.litellm.ai/docs/
    chat_response = litellm.completion(
        api_key=os.environ.get("OPENAI_API_KEY", "DUMMY_API_KEY"),
        model="gpt-3.5-turbo-0125",
        messages=[{"content": "Hello, how are you?", "role": "user"}],
    )

    all_content = chat_response.choices[0].message.content
    exp = """Hello! I'm just a computer program, so I don't have feelings, but I'm here to help you. How can I assist you today?"""

    assert all_content == exp
    calls = list(client.get_calls())
    assert len(calls) == 2
    call = calls[0]
    assert call.exception is None
    assert call.ended_at is not None
    output = call.output
    assert output["choices"][0]["message"]["content"] == exp
    assert output["choices"][0]["finish_reason"] == "stop"
    assert output["id"] == chat_response.id
    assert output["model"] == chat_response.model
    assert output["object"] == chat_response.object
    assert output["created"] == Nearly(chat_response.created)
    summary = call.summary
    assert summary is not None
    model_usage = summary["usage"][output["model"]]
    assert model_usage["requests"] == 1
    assert (
        output["usage"]["completion_tokens"] == model_usage["completion_tokens"] == 31
    )
    assert output["usage"]["prompt_tokens"] == model_usage["prompt_tokens"] == 13
    assert output["usage"]["total_tokens"] == model_usage["total_tokens"] == 44


@pytest.mark.skip_clickhouse_client  # TODO:VCR recording does not seem to allow us to make requests to the clickhouse db in non-recording mode
@pytest.mark.vcr(
    filter_headers=["authorization"], allowed_hosts=["api.wandb.ai", "localhost"]
)
@pytest.mark.asyncio
async def test_litellm_quickstart_async(
    client: weave.trace.weave_client.WeaveClient, patch_litellm: None
) -> None:
    # This is taken directly from https://docs.litellm.ai/docs/
    chat_response = await litellm.acompletion(
        api_key=os.environ.get("OPENAI_API_KEY", "DUMMY_API_KEY"),
        model="gpt-3.5-turbo-0125",
        messages=[{"content": "Hello, how are you?", "role": "user"}],
    )

    all_content = chat_response.choices[0].message.content
    exp = """Hello! I'm just a computer program, so I don't have feelings, but I'm here to help you with whatever you need. How can I assist you today?"""

    assert all_content == exp
    calls = list(client.get_calls())
    assert len(calls) == 2
    call = calls[0]
    assert call.exception is None
    assert call.ended_at is not None
    output = call.output
    assert output["choices"][0]["message"]["content"] == exp
    assert output["choices"][0]["finish_reason"] == "stop"
    assert output["id"] == chat_response.id
    assert output["model"] == chat_response.model
    assert output["object"] == chat_response.object
    assert output["created"] == Nearly(chat_response.created)
    summary = call.summary
    assert summary is not None

    model_usage = summary["usage"][output["model"]]
    assert model_usage["requests"] == 1
    assert (
        output["usage"]["completion_tokens"] == model_usage["completion_tokens"] == 35
    )
    assert output["usage"]["prompt_tokens"] == model_usage["prompt_tokens"] == 13
    assert output["usage"]["total_tokens"] == model_usage["total_tokens"] == 48


@pytest.mark.skip_clickhouse_client  # TODO:VCR recording does not seem to allow us to make requests to the clickhouse db in non-recording mode
@pytest.mark.vcr(
    filter_headers=["authorization"], allowed_hosts=["api.wandb.ai", "localhost"]
)
def test_litellm_quickstart_stream(
    client: weave.trace.weave_client.WeaveClient, patch_litellm: None
) -> None:
    # This is taken directly from https://docs.litellm.ai/docs/
    chat_response = litellm.completion(
        api_key=os.environ.get("OPENAI_API_KEY", "DUMMY_API_KEY"),
        model="gpt-3.5-turbo-0125",
        messages=[{"content": "Hello, how are you?", "role": "user"}],
        stream=True,
    )

    all_content = ""
    for chunk in chat_response:
        if chunk.choices[0].delta.content:
            all_content += chunk.choices[0].delta.content

    exp = """Hello! I'm just a computer program, so I don't have feelings, but I'm here to help you. How can I assist you today?"""

    assert all_content == exp
    calls = list(client.get_calls())
    assert len(calls) == 2
    call = calls[0]
    assert call.exception is None
    assert call.ended_at is not None
    output = call.output
    assert output["choices"][0]["message"]["content"] == exp
    assert output["choices"][0]["finish_reason"] == "stop"
    assert output["id"] == chunk.id
    assert output["model"] == chunk.model
    assert output["created"] == Nearly(chunk.created)
    summary = call.summary
    assert summary is not None

    # We are stuck here:
    # 1. LiteLLM uses raw responses, which we can't wrap in our iterator
    # 2. They don't even capture token usage correctly, so this info is
    # not available for now.
    if not USES_RAW_OPENAI_RESPONSE:
        model_usage = summary["usage"][output["model"]]
        assert model_usage["requests"] == 1
        assert model_usage["completion_tokens"] == 31
        assert model_usage["prompt_tokens"] == 13
        assert model_usage["total_tokens"] == 44


@pytest.mark.skip_clickhouse_client  # TODO:VCR recording does not seem to allow us to make requests to the clickhouse db in non-recording mode
@pytest.mark.vcr(
    filter_headers=["authorization"], allowed_hosts=["api.wandb.ai", "localhost"]
)
@pytest.mark.asyncio
async def test_litellm_quickstart_stream_async(
    client: weave.trace.weave_client.WeaveClient, patch_litellm: None
) -> None:
    # This is taken directly from https://docs.litellm.ai/docs/
    chat_response = await litellm.acompletion(
        api_key=os.environ.get("OPENAI_API_KEY", "DUMMY_API_KEY"),
        model="gpt-3.5-turbo-0125",
        messages=[{"content": "Hello, how are you?", "role": "user"}],
        stream=True,
    )

    all_content = ""
    async for chunk in chat_response:
        if chunk.choices[0].delta.content:
            all_content += chunk.choices[0].delta.content
    exp = """Hello! I'm just a computer program, so I don't have feelings, but I'm here and ready to assist you with any questions or tasks you may have. How can I help you today?"""

    assert all_content == exp
    calls = list(client.get_calls())
    assert len(calls) == 2
    call = calls[0]
    assert call.exception is None
    assert call.ended_at is not None
    output = call.output
    assert output["choices"][0]["message"]["content"] == exp
    assert output["choices"][0]["finish_reason"] == "stop"
    assert output["id"] == chunk.id
    assert output["model"] == chunk.model
    assert output["created"] == Nearly(chunk.created)
    summary = call.summary
    assert summary is not None

    # We are stuck here:
    # 1. LiteLLM uses raw responses, which we can't wrap in our iterator
    # 2. They don't even capture token usage correctly, so this info is
    # not available for now.
    if not USES_RAW_OPENAI_RESPONSE:
        model_usage = summary["usage"][output["model"]]
        assert model_usage["requests"] == 1
        assert model_usage["completion_tokens"] == 41
        assert model_usage["prompt_tokens"] == 13
        assert model_usage["total_tokens"] == 54


@pytest.mark.skip_clickhouse_client  # TODO:VCR recording does not seem to allow us to make requests to the clickhouse db in non-recording mode
@pytest.mark.vcr(
    filter_headers=["authorization", "x-api-key"],
    allowed_hosts=["api.wandb.ai", "localhost"],
)
def test_model_predict(
    client: weave.trace.weave_client.WeaveClient, patch_litellm: None
) -> None:
    class TranslatorModel(weave.Model):
        model: str
        temperature: float

        @weave.op
        def predict(self, text: str, target_language: str) -> str:
            response = litellm.completion(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-ant-DUMMY_API_KEY"),
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a translator. Translate the given text to {target_language}.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=1024,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

    # Create instances with different models
    claude_translator = TranslatorModel(
        model="claude-3-5-sonnet-20240620", temperature=0.1
    )

    res = claude_translator.predict("There is a bug in my code!", "Spanish")
    assert res is not None

    call = claude_translator.predict.calls()[0]
    d = call.summary["usage"]["claude-3-5-sonnet-20240620"]
    assert d["cache_creation_input_tokens"] == 0
    assert d["cache_read_input_tokens"] == 0
    assert d["requests"] == 1
    assert d["prompt_tokens"] == 28
    assert d["prompt_tokens_details"]["cached_tokens"] == 0
    assert d["completion_tokens"] == 10
    assert d["total_tokens"] == 38


def _make_streaming_chunk(
    chunk_id: str,
    model: str,
    content: str | None = None,
    reasoning_content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
    choice_index: int = 0,
) -> ModelResponse:
    """Helper to create a streaming ModelResponse chunk for accumulator tests."""
    delta_kwargs: dict[str, Any] = {}
    if content is not None:
        delta_kwargs["content"] = content
    if reasoning_content is not None:
        delta_kwargs["reasoning_content"] = reasoning_content
    if role is not None:
        delta_kwargs["role"] = role

    return ModelResponse(
        id=chunk_id,
        object="chat.completion.chunk",
        created=1234567890,
        model=model,
        choices=[
            StreamingChoices(
                index=choice_index,
                delta=Delta(**delta_kwargs),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(prompt_tokens=0, total_tokens=0, completion_tokens=None),
    )


def test_litellm_accumulator_reasoning_content() -> None:
    """Test that reasoning_content from streaming deltas is accumulated correctly."""
    chunks = [
        _make_streaming_chunk("id1", "deepseek-r1", role="assistant", reasoning_content="Let me "),
        _make_streaming_chunk("id1", "deepseek-r1", reasoning_content="think about "),
        _make_streaming_chunk("id1", "deepseek-r1", reasoning_content="this..."),
        _make_streaming_chunk("id1", "deepseek-r1", content="The answer is 4."),
        _make_streaming_chunk("id1", "deepseek-r1", finish_reason="stop"),
    ]

    acc = None
    for chunk in chunks:
        acc = litellm_accumulator(acc, chunk)

    assert acc is not None
    assert len(acc.choices) == 1
    assert acc.choices[0].message.content == "The answer is 4."
    assert hasattr(acc.choices[0].message, "reasoning_content")
    assert acc.choices[0].message.reasoning_content == "Let me think about this..."
    assert acc.choices[0].finish_reason == "stop"


def test_litellm_accumulator_reasoning_content_interleaved_with_content() -> None:
    """Test accumulation when reasoning_content and content arrive in the same chunks."""
    chunks = [
        _make_streaming_chunk("id1", "deepseek-r1", role="assistant", reasoning_content="Step 1: ", content=""),
        _make_streaming_chunk("id1", "deepseek-r1", reasoning_content="2+2=4", content="The "),
        _make_streaming_chunk("id1", "deepseek-r1", content="answer is 4."),
        _make_streaming_chunk("id1", "deepseek-r1", finish_reason="stop"),
    ]

    acc = None
    for chunk in chunks:
        acc = litellm_accumulator(acc, chunk)

    assert acc is not None
    assert acc.choices[0].message.content == "The answer is 4."
    assert acc.choices[0].message.reasoning_content == "Step 1: 2+2=4"


def test_litellm_accumulator_no_reasoning_content() -> None:
    """Test that the accumulator works correctly when no reasoning_content is present."""
    chunks = [
        _make_streaming_chunk("id1", "gpt-4", role="assistant", content="Hello "),
        _make_streaming_chunk("id1", "gpt-4", content="world!"),
        _make_streaming_chunk("id1", "gpt-4", finish_reason="stop"),
    ]

    acc = None
    for chunk in chunks:
        acc = litellm_accumulator(acc, chunk)

    assert acc is not None
    assert acc.choices[0].message.content == "Hello world!"
    # reasoning_content should not be set when no reasoning chunks were received
    assert not hasattr(acc.choices[0].message, "reasoning_content")
