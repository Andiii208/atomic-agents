"""Tests for AgentConfig mode resolution and client/config mode consistency (issue #282).

The mode on the Instructor client decides the API call format, while
``AgentConfig.mode`` only drives token accounting. These tests cover deriving the
mode from the client, warning when an explicit mode disagrees with the client on
how the schema is transmitted, and keeping the accounting aligned with the
requests Instructor actually prepares.
"""

import logging
from unittest.mock import Mock

import instructor
import pytest
from instructor.processing.response import handle_response_model
from openai import OpenAI
from pydantic import Field

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.agents.atomic_agent import _TOOL_MODES
from atomic_agents.context import ChatHistory

DUMMY_KEY = "not-a-real-key"
LOGGER_NAME = "atomic_agents.agents.atomic_agent"
COUNTING_MODEL = "gpt-4o-mini"
TOOL_REQUEST_PARAMS = ("tools", "functions", "toolConfig")


class InputWithImage(BaseIOSchema):
    """Input with image."""

    chat_message: str = Field(..., description="Text input")
    image: instructor.Image = Field(..., description="Image to analyze")


def _client(mode):
    """Build a real Instructor client with the given mode (no request is ever made)."""
    return instructor.from_openai(OpenAI(api_key=DUMMY_KEY, base_url="https://example.invalid/v1"), mode=mode)


def _mock_client(mode):
    """A client double that exposes `mode` like a real Instructor client does."""
    client = Mock(spec=instructor.Instructor)
    client.mode = mode
    return client


def _agent(config):
    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)


def _history_with_image():
    history = ChatHistory()
    history.add_message(
        "user", InputWithImage(chat_message="Look", image=instructor.Image.from_url("https://example.com/x.png"))
    )
    return history


def _prepared_request(mode, messages=None):
    """Return the kwargs Instructor would send for `mode`, prepared without any provider call."""
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return handle_response_model(
        response_model=BasicChatOutputSchema,
        mode=mode,
        messages=[dict(message) for message in messages],
        model="test-model",
    )[1]


class TestModeDerivation:
    """AgentConfig.mode defaults to the client's mode."""

    def test_none_derives_client_mode(self):
        agent = _agent(AgentConfig(client=_client(instructor.Mode.JSON), model=COUNTING_MODEL))

        assert agent.mode == instructor.Mode.JSON

    def test_none_derives_tools_mode(self):
        agent = _agent(AgentConfig(client=_client(instructor.Mode.TOOLS), model=COUNTING_MODEL))

        assert agent.mode == instructor.Mode.TOOLS

    def test_none_derives_native_provider_mode(self):
        agent = _agent(AgentConfig(client=_mock_client(instructor.Mode.ANTHROPIC_TOOLS), model=COUNTING_MODEL))

        assert agent.mode == instructor.Mode.ANTHROPIC_TOOLS

    def test_derived_mode_drives_token_accounting(self):
        json_agent = _agent(AgentConfig(client=_client(instructor.Mode.JSON), model=COUNTING_MODEL))
        tools_agent = _agent(AgentConfig(client=_client(instructor.Mode.TOOLS), model=COUNTING_MODEL))

        assert json_agent._build_tools_definition() is None
        assert tools_agent._build_tools_definition() is not None

    def test_derived_native_mode_counts_schema_as_tools(self):
        agent = _agent(AgentConfig(client=_mock_client(instructor.Mode.ANTHROPIC_TOOLS), model=COUNTING_MODEL))

        assert agent._build_tools_definition() is not None
        assert agent.get_context_token_count().tools > 0


class TestClientWithoutMode:
    """Clients that expose no Mode attribute fall back to TOOLS without warnings."""

    def test_mock_client_falls_back_to_tools_mode(self, caplog):
        mock_client = Mock(spec=instructor.Instructor)  # spec exposes no instance attribute `mode`
        agent = _agent(AgentConfig(client=mock_client, model=COUNTING_MODEL))

        assert agent.mode == instructor.Mode.TOOLS
        assert not [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno >= logging.WARNING]

    def test_explicit_mode_is_kept_for_mock_client(self, caplog):
        mock_client = Mock(spec=instructor.Instructor)
        agent = _agent(AgentConfig(client=mock_client, model=COUNTING_MODEL, mode=instructor.Mode.JSON))

        assert agent.mode == instructor.Mode.JSON
        assert not [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno >= logging.WARNING]


class TestModeConsistencyWarning:
    """Explicit modes are checked against how the client transmits the schema."""

    def test_matching_mode_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            agent = _agent(AgentConfig(client=_client(instructor.Mode.JSON), model=COUNTING_MODEL, mode=instructor.Mode.JSON))

        assert agent.mode == instructor.Mode.JSON
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]

    def test_mismatching_accounting_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            agent = _agent(AgentConfig(client=_client(instructor.Mode.JSON), model=COUNTING_MODEL, mode=instructor.Mode.TOOLS))

        assert agent.mode == instructor.Mode.TOOLS
        warnings = [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "TOOLS" in warnings[0].getMessage()
        assert "JSON" in warnings[0].getMessage()

    def test_same_accounting_different_mode_does_not_warn(self, caplog):
        # TOOLS_STRICT on the config and TOOLS on the client both send a tools definition.
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            agent = _agent(
                AgentConfig(client=_client(instructor.Mode.TOOLS), model=COUNTING_MODEL, mode=instructor.Mode.TOOLS_STRICT)
            )

        assert agent.mode == instructor.Mode.TOOLS_STRICT
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]

    def test_native_provider_mode_matches_tools_accounting(self, caplog):
        # ANTHROPIC_TOOLS on the client and TOOLS on the config are counted the same way.
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            agent = _agent(
                AgentConfig(
                    client=_mock_client(instructor.Mode.ANTHROPIC_TOOLS), model=COUNTING_MODEL, mode=instructor.Mode.TOOLS
                )
            )

        assert agent.mode == instructor.Mode.TOOLS
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]

    def test_native_provider_mode_against_json_accounting_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _agent(
                AgentConfig(
                    client=_mock_client(instructor.Mode.ANTHROPIC_TOOLS), model=COUNTING_MODEL, mode=instructor.Mode.JSON
                )
            )

        message = [r for r in caplog.records if r.name == LOGGER_NAME][0].getMessage()
        assert "AgentConfig.mode (JSON)" in message
        assert "client's mode (ANTHROPIC_TOOLS)" in message

    def test_warning_mentions_both_directions(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _agent(AgentConfig(client=_client(instructor.Mode.TOOLS), model=COUNTING_MODEL, mode=instructor.Mode.JSON))

        message = [r for r in caplog.records if r.name == LOGGER_NAME][0].getMessage()
        assert "AgentConfig.mode (JSON)" in message
        assert "client's mode (TOOLS)" in message


class TestMultimodalTokenCounting:
    """Media serialization stays in the chat format the token counter accepts."""

    def test_image_history_counts_in_responses_mode(self):
        # Regression check: Responses-format input_image parts used to reach the
        # chat token counter and make get_context_token_count() raise.
        agent = _agent(
            AgentConfig(client=_client(instructor.Mode.RESPONSES_TOOLS), model=COUNTING_MODEL, history=_history_with_image())
        )

        result = agent.get_context_token_count()

        assert result.history > 0
        assert result.total > 0

    def test_media_parts_stay_chat_format_in_responses_mode(self):
        agent = _agent(
            AgentConfig(client=_client(instructor.Mode.RESPONSES_TOOLS), model=COUNTING_MODEL, history=_history_with_image())
        )

        serialized = agent._serialize_history_for_token_count()

        assert [part["type"] for part in serialized[0]["content"]] == ["text", "image_url"]

    def test_image_history_counts_in_tools_mode(self):
        agent = _agent(AgentConfig(client=_client(instructor.Mode.TOOLS), model=COUNTING_MODEL, history=_history_with_image()))

        result = agent.get_context_token_count()

        assert result.history > 0
        assert result.tools > 0


class TestAccountingMatchesPreparedRequests:
    """The schema must be counted the same way Instructor transmits it."""

    @pytest.mark.parametrize("mode", list(instructor.Mode), ids=lambda mode: mode.name)
    def test_tools_accounting_matches_prepared_request(self, mode):
        try:
            kwargs = _prepared_request(mode)
        except Exception as e:
            # Preparing some modes needs their provider SDK (not installed here) or an
            # Iterable response model (parallel tool calls); those cannot be checked
            # offline, and the accounting set covers them by construction.
            pytest.skip(f"{mode.name} cannot be prepared without its provider setup: {type(e).__name__}: {e}")

        sends_tool_definition = any(param in kwargs for param in TOOL_REQUEST_PARAMS)

        assert sends_tool_definition == (mode in _TOOL_MODES)

    def test_anthropic_tools_request_carries_tool_without_schema_prose(self):
        kwargs = _prepared_request(
            instructor.Mode.ANTHROPIC_TOOLS,
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "hello"}],
        )

        assert kwargs["tools"]
        assert kwargs["tool_choice"]
        assert not any("json_schema" in str(block.get("text", "")) for block in kwargs["system"])
        assert not any("json_schema" in str(message.get("content", "")) for message in kwargs["messages"])
