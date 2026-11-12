# SPDX-License-Identifier: Apache-2.0
"""CPU checks for compose.tools.yaml (upstream vLLM #62199, #62517, #62729, #64971 on image .11).

Run through ../check_tools.sh. Tests marked ``fixed_by_port(<pr>)`` encode an expectation added by the
ported PR; with DS41_TOOLS_EXPECT=unpatched they must FAIL on the stock image (strict xfail, see
conftest.py). Unmarked tests are regression checks that must pass on both.

The 1017 reference encoder is loaded from DS41_TOOLS_REF (default /ref/encoding.py, the checkpoint's
``encoding/`` dir mounted read-only); its golden fixtures from /ref/tests.
Several tests are adapted from upstream tests/tokenizers_/test_deepseek_v41.py and
tests/parser/engine/test_deepseek_v41.py (@ 0f0172d9).
"""

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.deepseek_v41 import deepseek_v41_config
from vllm.parser.parser_manager import ParserManager
from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer
from vllm.tokenizers.deepseek_v41_encoding import IMAGE_PLACEHOLDER, encode_messages

REF_PATH = Path(os.environ.get("DS41_TOOLS_REF", "/ref/encoding.py"))
REF_TESTS = REF_PATH.parent / "tests"
REF_TOKENIZER = Path(os.environ.get("DS41_TOOLS_REF_TOKENIZER", "/ref-tok/tokenizer.json"))

BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"


def _load_reference():
    spec = importlib.util.spec_from_file_location("ds41_ref_encoding", REF_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REF = _load_reference()


class FakeHfTokenizer:
    """Same stand-in as upstream tests/tokenizers_/test_deepseek_v4.py."""

    vocab_size = 100

    def get_added_vocab(self) -> dict[str, int]:
        return {"</think>": 100}

    def encode(self, text: str, add_special_tokens: bool = False, **kwargs) -> list[int]:
        self.last_encode = (text, add_special_tokens, kwargs)
        return [len(text)]


def render(messages, **kwargs):
    return get_deepseek_v41_tokenizer(FakeHfTokenizer()).apply_chat_template(
        messages, tokenize=False, **kwargs
    )


def opener(thinking: bool) -> str:
    return "<｜Assistant｜>" + ("<think>" if thinking else "</think>")


# ---------------------------------------------------------------- 1017 reference: golden fixtures


def _fixture(case_id):
    data = json.loads((REF_TESTS / f"test_input_{case_id}.json").read_text())
    case = data if isinstance(data, dict) else {"messages": data}
    expected = (REF_TESTS / f"test_output_{case_id}.txt").read_text()
    return case, expected


def _render_fixture(case, **extra):
    thinking_mode = case.get("thinking_mode") or "chat"
    return render(
        case["messages"],
        tools=case.get("tools"),
        thinking=thinking_mode == "thinking",
        reasoning_effort=case.get("reasoning_effort"),
        **extra,
    )


@pytest.mark.parametrize("case_id", [1, 2, 3, 4, 5])
def test_reference_golden_fixtures(case_id):
    """Closed history (add_generation_prompt=False when the last turn is assistant) = reference."""
    case, expected = _fixture(case_id)
    final_assistant = case["messages"][-1]["role"] == "assistant"
    assert _render_fixture(case, add_generation_prompt=not final_assistant) == expected


@pytest.mark.fixed_by_port(62729)
@pytest.mark.parametrize("case_id", [1, 2, 3, 4])
def test_reference_fixture_final_assistant_gets_new_turn_by_default(case_id):
    case, expected = _fixture(case_id)
    assert case["messages"][-1]["role"] == "assistant"
    thinking = (case.get("thinking_mode") or "chat") == "thinking"
    assert _render_fixture(case) == expected + opener(thinking)


# ---------------------------------------------------------------- multi-turn tool conversation vs reference

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city (Celsius).",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Local time for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def _call(call_id, name, city):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"city": city})},
    }


def _conversation(shape: str):
    msgs = [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "What's the weather and time in Paris? Answer in Fahrenheit."},
        {
            "role": "assistant",
            "content": "Let me check.",
            "reasoning": "Tool returns Celsius; convert to Fahrenheit. Need both tools.",
            "tool_calls": [_call("c1", "get_weather", "Paris"), _call("c2", "get_time", "Paris")],
        },
        # Results out of call order: the encoder sorts them by call order.
        {"role": "tool", "tool_call_id": "c2", "content": '{"time": "14:05"}'},
        {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 20}'},
    ]
    if shape in ("final", "followup"):
        msgs.append(
            {
                "role": "assistant",
                "content": "68°F and 14:05 in Paris.",
                "reasoning": "20 C = 68 F.",
            }
        )
    if shape == "followup":
        msgs.append({"role": "user", "content": "And in Tokyo?"})
    return msgs


def _reference(msgs, tools, thinking):
    ref_msgs = []
    for m in copy.deepcopy(msgs):
        if "reasoning" in m:
            m["reasoning_content"] = m.pop("reasoning")
        ref_msgs.append(m)
    if tools:
        ref_msgs[0]["tools"] = tools
    return REF.encode_messages(ref_msgs, thinking_mode="thinking" if thinking else "chat")


@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize("with_tools", [True, False])
@pytest.mark.parametrize("shape", ["after_tool", "final", "followup"])
def test_tool_conversation_matches_1017_reference(thinking, with_tools, shape):
    """user -> assistant(reasoning + 2 tool calls) -> tool results -> [assistant -> user]."""
    msgs = _conversation(shape)
    tools = TOOLS if with_tools else None
    expected = _reference(msgs, tools, thinking)
    got = render(
        msgs,
        tools=tools,
        thinking=thinking,
        add_generation_prompt=shape != "final",
    )
    assert got == expected
    # Tool results in call order, inside one user turn.
    assert got.index("<tool_result>{\"temp_c\": 20}") < got.index("<tool_result>{\"time\"")


@pytest.mark.parametrize("with_tools", [True, False])
def test_reasoning_before_tool_call_is_kept(with_tools):
    """#62199 core (tool-result-only user turns do not start a turn). Already in .11 via the BRRR
    tool-loop fix, so this is a regression check, not fixed_by_port."""
    got = render(_conversation("after_tool"), tools=TOOLS if with_tools else None)
    assert (
        "<｜Assistant｜><think>Tool returns Celsius; convert to Fahrenheit. Need both tools.</think>"
        "Let me check.\n\n<｜DSML｜ calls>" in got
    )
    assert got.endswith("</tool_result><｜Assistant｜><think>")
    # The next real user turn drops it again (no tools => drop_thinking applies).
    nxt = render(_conversation("followup"))
    assert "Tool returns Celsius" not in nxt and nxt.endswith("And in Tokyo?" + opener(True))


@pytest.mark.fixed_by_port(62729)
@pytest.mark.parametrize("thinking", [True, False])
def test_tool_conversation_final_assistant_default_opens_turn(thinking):
    msgs = _conversation("final")
    assert render(msgs, tools=TOOLS, thinking=thinking) == _reference(
        msgs, TOOLS, thinking
    ) + opener(thinking)


# ---------------------------------------------------------------- #62199: literal image token -> 400


@pytest.mark.fixed_by_port(62199)
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("as_part", [False, True])
def test_rejects_image_placeholder_in_text(role, as_part):
    text = f"hi {IMAGE_PLACEHOLDER}"
    content = [{"type": "text", "text": text}] if as_part else text
    with pytest.raises(ValueError, match="image special token"):
        render([{"role": role, "tool_call_id": "c1", "content": content}])


@pytest.mark.fixed_by_port(62199)
def test_rejects_image_placeholder_in_reasoning():
    with pytest.raises(ValueError, match="image special token"):
        render(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning": IMAGE_PLACEHOLDER},
            ]
        )


def test_image_parts_still_render():
    got = render(
        [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "x"}}]}],
        thinking=False,
    )
    assert got == BOS + "<｜User｜>a\n\n" + IMAGE_PLACEHOLDER + "<｜Assistant｜></think>"


# ---------------------------------------------------------------- #62729: generation controls


@pytest.mark.fixed_by_port(62729)
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("final_role", ["user", "developer", "system", "assistant"])
def test_generation_prompt_only_controls_final_turn(thinking, final_role):
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": final_role, "content": "final message"},
    ]
    original = copy.deepcopy(messages)
    closed = render(messages, thinking=thinking, add_generation_prompt=False)
    opened = render(messages, thinking=thinking, add_generation_prompt=True)
    assert opened == closed + opener(thinking)
    assert "first answer" + EOS in closed
    assert "<｜User｜>first question<｜Assistant｜>" in closed
    assert messages == original


def test_default_user_final_unchanged():
    assert render([{"role": "user", "content": "q"}], thinking=False) == (
        BOS + "<｜User｜>q<｜Assistant｜></think>"
    )


@pytest.mark.fixed_by_port(62729)
@pytest.mark.parametrize("thinking", [False, True])
def test_continue_final_message_preserves_historical_eos(thinking):
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "The answer is "},
    ]
    closed = render(messages, thinking=thinking, add_generation_prompt=False)
    continued = render(
        messages, thinking=thinking, add_generation_prompt=False, continue_final_message=True
    )
    assert closed == continued + EOS
    assert continued.endswith("The answer is ")
    assert "first answer" + EOS in continued


@pytest.mark.fixed_by_port(62729)
def test_continue_final_message_through_chat_request():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    request = ChatCompletionRequest(
        model="DeepSeek-V4.1-Flash",
        messages=[
            {"role": "user", "content": "Finish this sentence."},
            {"role": "assistant", "content": "The answer is "},
        ],
        continue_final_message=True,
        add_generation_prompt=False,
    )
    kwargs = request.build_chat_params(None, "auto").get_apply_chat_template_kwargs()
    assert render(request.messages, **kwargs).endswith("The answer is ")


@pytest.mark.fixed_by_port(62729)
def test_continue_final_reasoning_keeps_thinking_open():
    messages = [
        {"role": "user", "content": "Explain your answer."},
        {"role": "assistant", "content": None, "reasoning": "Let me consider "},
    ]
    assert render(messages, continue_final_message=True).endswith(
        "<｜Assistant｜><think>Let me consider "
    )


@pytest.mark.parametrize("thinking_mode", ["chat", "thinking"])
@pytest.mark.parametrize(
    "final",
    [
        {"content": "The answer is "},
        {"content": "", "reasoning_content": "Let me consider "},
        {"content": "", "tool_calls": [_call("c1", "get_weather", "Paris")]},
    ],
)
def test_open_final_message_gets_no_generation_prompt(thinking_mode, final):
    """wo_eos final assistant (encoder-internal, no request can set it): no EOS and no new assistant
    turn by default, i.e. it renders exactly as on the stock image."""
    messages = [{"role": "user", "content": "question"}, {"role": "assistant", "wo_eos": True, **final}]
    closed = encode_messages(copy.deepcopy(messages), thinking_mode=thinking_mode)
    assert not closed.endswith(EOS)
    assert closed.count("<｜Assistant｜>") == 1


@pytest.mark.fixed_by_port(62729)
@pytest.mark.parametrize("thinking_mode", ["chat", "thinking"])
def test_open_final_message_ignores_add_generation_prompt(thinking_mode):
    messages = [{"role": "user", "content": "question"}, {"role": "assistant", "wo_eos": True, "content": "x"}]

    def encode(**kwargs):
        return encode_messages(copy.deepcopy(messages), thinking_mode=thinking_mode, **kwargs)

    assert encode() == encode(add_generation_prompt=False)


@pytest.mark.fixed_by_port(62729)
@pytest.mark.parametrize("messages", [[], [{"role": "user", "content": "hello"}]])
def test_continue_final_message_requires_assistant(messages):
    with pytest.raises(ValueError, match="final assistant"):
        render(messages, continue_final_message=True)


# ---------------------------------------------------------------- #64971: tool_reference


@pytest.mark.fixed_by_port(64971)
def test_tool_references_render_as_tool_names():
    messages = [
        {"role": "assistant", "tool_calls": [_call("tool_search", "tool_search", "x")]},
        {
            "role": "tool",
            "tool_call_id": "tool_search",
            "content": [
                {"type": "tool_reference", "name": "get_weather"},
                {"type": "tool_reference", "name": "get_time"},
            ],
        },
    ]
    assert render(messages, thinking=False).endswith(
        "<｜User｜><tool_result>get_weather\n\nget_time</tool_result><｜Assistant｜></think>"
    )


def test_unsupported_media_is_still_a_request_error():
    with pytest.raises(ValueError, match="text and image content only"):
        render([{"role": "user", "content": [{"type": "input_audio"}]}])


def _anthropic(messages, **kwargs):
    from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    request = AnthropicMessagesRequest(model="DeepSeek-V4.1-Flash", max_tokens=16, messages=messages, **kwargs)
    return AnthropicServingMessages.to_chat_completion_request(request)


@pytest.mark.fixed_by_port(64971)
def test_anthropic_tool_result_with_tool_references_is_one_tool_message():
    result = _anthropic(
        [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_001", "name": "read_file", "input": {"path": "/tmp/x"}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_001",
                        "content": [
                            {"type": "text", "text": "2 tools found"},
                            {"type": "tool_reference", "tool_name": "get_weather"},
                            {"type": "tool_reference", "tool_name": "get_time"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                        ],
                    }
                ],
            },
        ]
    )
    assert [m["role"] for m in result.messages] == ["assistant", "tool", "user"]
    assert result.messages[1] == {
        "role": "tool",
        "tool_call_id": "call_001",
        "content": [
            {"type": "tool_reference", "name": "get_weather"},
            {"type": "tool_reference", "name": "get_time"},
            {"type": "text", "text": "2 tools found"},
        ],
    }


@pytest.mark.fixed_by_port(64971)
def test_anthropic_tool_search_renders_one_result_per_call_on_v41():
    result = _anthropic(
        [
            {"role": "user", "content": "Read notes.txt and find a weather tool."},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "tool_search", "input": {"query": "weather"}},
                    {"type": "tool_use", "id": "toolu_2", "name": "read_file", "input": {"path": "notes.txt"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "tool_reference", "tool_name": "get_weather"}]},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "notes"},
                ],
            },
        ]
    )
    got = render([dict(m) for m in result.messages], thinking=False)
    assert got.endswith(
        "<｜User｜><tool_result>get_weather</tool_result>\n\n<tool_result>notes</tool_result><｜Assistant｜></think>"
    )


def test_anthropic_plain_tool_result_unchanged():
    result = _anthropic(
        [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "f", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "ok"}]},
        ]
    )
    assert result.messages[1] == {"role": "tool", "tool_call_id": "c", "content": "ok"}


# ---------------------------------------------------------------- parser (#62517, #62729)

CALLS = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">杭州</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="count" string="false">42</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n<｜DSML｜ invoke name="add">\n'
    '<｜DSML｜ parameter name="x" string="false">1.5</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="y" string="false">2.25</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)
V4_NAMED_CALLS = CALLS.replace("｜DSML｜ calls>", "｜DSML｜ tool_calls>")
EXPECTED_ARGS = [{"city": "杭州", "count": 42}, {"x": 1.5, "y": 2.25}]


def tokenizer_for(text, special_calls=False):
    vocab = {"<think>": 50, "</think>": 51, "<｜User｜>": 54, "<｜Assistant｜>": 55}
    if special_calls:
        vocab |= {"<｜DSML｜ calls>": 52, "</｜DSML｜ calls>": 53}
    tokens: list[tuple[int, str]] = []
    while text:
        special = next((marker for marker in vocab if text.startswith(marker)), None)
        piece = special or text[0]
        tokens.append((vocab[piece] if special else 100 + len(tokens), piece))
        text = text[len(piece) :]
    return MockTokenizer(vocab, tokens), tokens


def parser_for(tokenizer, controls):
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name="deepseek_v41",
        enable_auto_tools=True,
    )
    assert cls is not None
    return cls(tokenizer, chat_template_kwargs=controls)


def _stream(text, thinking, chunk_size, special_calls=False):
    tokenizer, tokens = tokenizer_for(text, special_calls)
    return collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": thinking}),
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )


def _full(text, thinking):
    tokenizer, tokens = tokenizer_for(text)
    return parser_for(tokenizer, {"thinking": thinking}).parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("special_calls", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_parser_canonical_calls(thinking, special_calls, chunk_size):
    text = ("Plan.</think>" if thinking else "") + "Checking." + CALLS
    out = _stream(text, thinking, chunk_size, special_calls)
    assert out.reasoning == ("Plan." if thinking else "")
    assert out.content.strip() == "Checking."
    assert [c["name"] for c in out.tool_calls] == ["get_weather", "add"]
    assert [json.loads(c["arguments"]) for c in out.tool_calls] == EXPECTED_ARGS
    reasoning, content, calls = _full(text, thinking)
    assert (reasoning or "") == ("Plan." if thinking else "")
    assert (content or "").strip() == "Checking."
    assert [json.loads(c.arguments) for c in calls] == EXPECTED_ARGS


@pytest.mark.parametrize("thinking", [False, True])
def test_parser_plain_answer_no_tools(thinking):
    text = ("Plan.</think>" if thinking else "") + "Just an answer."
    out = _stream(text, thinking, 1)
    assert out.content == "Just an answer." and not out.tool_calls


@pytest.mark.fixed_by_port(62517)
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_parser_v4_wrapper_name_opens_tool_block(thinking, chunk_size):
    text = ("Plan.</think>" if thinking else "") + "Checking." + V4_NAMED_CALLS
    out = _stream(text, thinking, chunk_size)
    reasoning, content, calls = _full(text, thinking)
    assert out.reasoning == (reasoning or "") == ("Plan." if thinking else "")
    assert out.content.strip() == (content or "").strip() == "Checking."
    assert "DSML" not in out.content and "DSML" not in (content or "")
    assert [c["name"] for c in out.tool_calls] == [c.name for c in calls] == ["get_weather", "add"]
    assert [json.loads(c["arguments"]) for c in out.tool_calls] == EXPECTED_ARGS


@pytest.mark.fixed_by_port(62517)
def test_parser_v4_wrapper_name_ends_reasoning():
    out = _stream("Plan." + V4_NAMED_CALLS, True, 1)
    assert out.reasoning == "Plan."
    assert not out.content.strip()
    assert [c["name"] for c in out.tool_calls] == ["get_weather", "add"]


def test_parser_unspaced_v4_block_is_still_text():
    text = CALLS.replace("｜DSML｜ ", "｜DSML｜").replace("｜DSML｜calls>", "｜DSML｜tool_calls>")
    out = _stream(text, False, 1)
    assert not out.tool_calls


# Streaming only: the image's non-streaming parse() has no prompt hook (upstream set_prompt_token_ids
# came later); turn boundaries reach streaming and the engine-side reasoning_ended check
# (chat_completion/serving.py: parser.is_reasoning_end(prompt_token_ids)).
@pytest.mark.parametrize("streaming", [True])
@pytest.mark.parametrize(
    ("prompt_token_ids", "text", "expected_reasoning", "expected_content"),
    [
        # continue_final_message: the current turn already closed reasoning.
        ([54, 100, 55, 50, 101, 51, 102], "12", "", "12"),
        # A closed block in an earlier turn does not end this turn's reasoning.
        ([54, 100, 55, 50, 101, 51, 102, 54, 103, 55, 50], "Plan.</think>12", "Plan.", "12"),
    ],
)
def test_prompt_closed_reasoning_is_scoped_to_current_turn(
    streaming, prompt_token_ids, text, expected_reasoning, expected_content
):
    tokenizer, tokens = tokenizer_for(text)
    parser = parser_for(tokenizer, {"thinking": True})
    output_ids = [tid for tid, _ in tokens]
    if streaming:
        out = collect_output(
            replay_streaming(parser, tokens, chunk_size=1, finished_on_last=True, prompt_token_ids=prompt_token_ids)
        )
        reasoning, content = out.reasoning, out.content
    else:
        parser.set_prompt_token_ids(prompt_token_ids)
        reasoning, content, _ = parser.parse(text, _test_request(), model_output_token_ids=output_ids)
    assert (reasoning or "") == expected_reasoning
    assert content == expected_content


@pytest.mark.fixed_by_port(62729)
def test_turn_boundaries_declared():
    assert deepseek_v41_config(thinking=True).turn_boundary_tokens == frozenset(
        {"<｜User｜>", "<｜Assistant｜>"}
    )


@pytest.mark.fixed_by_port(62729)
def test_earlier_turn_think_end_does_not_end_current_reasoning():
    """Prompt whose current turn has no think marker (template that skips thinking): only the earlier
    turn's </think> exists. With boundaries the scan stops at <｜Assistant｜> and the turn's reasoning
    state comes from the parser mode, not from the stale </think>."""
    tokenizer, _ = tokenizer_for("x")
    parser = parser_for(tokenizer, {"thinking": True})
    assert not parser.is_reasoning_end([54, 100, 55, 50, 101, 51, 102, 54, 103, 55])


# ---------------------------------------------------------------- real tokenizer sanity


def test_real_vocab_has_single_token_turn_markers():
    if not REF_TOKENIZER.exists():
        pytest.skip(f"{REF_TOKENIZER} not mounted")
    added = {t["content"] for t in json.loads(REF_TOKENIZER.read_text())["added_tokens"]}
    assert {"<｜User｜>", "<｜Assistant｜>", "<think>", "</think>"} <= added
    # The DSML openers are plain text (only ｜DSML｜ is an added token), so the parser matches them as text.
    assert "<｜DSML｜ calls>" not in added and "<｜DSML｜ tool_calls>" not in added


# ---------------------------------------------------------------- BRRR renderer reorder (baked, not overlaid)


def _brrr_renderer():
    from types import SimpleNamespace

    from vllm.renderers.registry import RENDERER_REGISTRY

    cls = RENDERER_REGISTRY.load_renderer_cls("deepseek_v41")
    renderer = cls.__new__(cls)
    renderer.model_config = SimpleNamespace(
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_prompt_embeds=False,
        is_multimodal_model=False,
        hf_config=SimpleNamespace(),
        tokenizer_mode="deepseek_v41",
    )
    renderer.tokenizer = get_deepseek_v41_tokenizer(FakeHfTokenizer())
    return renderer


def _renderer_conversation(with_reference):
    second = (
        [{"type": "tool_reference", "name": "get_weather"}, {"type": "text", "text": "1 found"}]
        if with_reference
        else "1 found"
    )
    return [
        {"role": "user", "content": "find tools"},
        {
            "role": "assistant",
            "content": None,
            "reasoning": "search first",
            "tool_calls": [_call("t1", "tool_search", "x"), _call("t2", "read_file", "y")],
        },
        {"role": "tool", "tool_call_id": "t2", "content": "notes"},
        {"role": "tool", "tool_call_id": "t1", "content": second},
    ]


def _check_renderer_path(with_reference):
    from vllm.renderers.params import ChatParams

    renderer = _brrr_renderer()
    msgs = _renderer_conversation(with_reference)
    for thinking in (True, False):
        _, prompt = renderer.render_messages(
            msgs, ChatParams(chat_template_kwargs={"tokenize": False, "thinking": thinking})
        )
        assert prompt["prompt"] == render(msgs, thinking=thinking)
        assert prompt["prompt"].index("<tool_result>") < prompt["prompt"].index("<tool_result>notes")
        assert prompt["prompt"].endswith("<tool_result>notes</tool_result>" + opener(thinking))


def test_brrr_renderer_reorder_path_matches_encoder():
    """The baked BRRR renderer (reorders raw messages into prompt order) + overlay encoder agree."""
    _check_renderer_path(with_reference=False)


@pytest.mark.fixed_by_port(64971)
def test_brrr_renderer_reorder_path_with_tool_reference():
    _check_renderer_path(with_reference=True)


@pytest.mark.fixed_by_port(62729)
def test_brrr_renderer_continue_final_message():
    from vllm.renderers.params import ChatParams

    msgs = _renderer_conversation(False) + [{"role": "assistant", "content": "The ans"}]
    _, prompt = _brrr_renderer().render_messages(
        msgs,
        ChatParams(
            chat_template_kwargs={"tokenize": False, "continue_final_message": True, "add_generation_prompt": False}
        ),
    )
    assert prompt["prompt"].endswith("<｜Assistant｜><think></think>The ans")
