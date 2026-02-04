from __future__ import annotations

from typing import Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .localai_client import LocalAIClient


class AgentState(TypedDict, total=False):
    model: str
    messages: List[Dict[str, str]]
    temperature: float
    stream: bool
    stream_callback: Optional[Callable[[str], None]]
    cancel_check: Optional[Callable[[], bool]]
    response: str


class SkippyAgentGraph:
    def __init__(self, client: LocalAIClient) -> None:
        self._client = client
        self._graph = self._build_graph()

    def _build_graph(self):
        def call_llm(state: AgentState) -> AgentState:
            model = state.get("model", "")
            messages = state.get("messages", [])
            temperature = state.get("temperature", 0.7)
            stream = state.get("stream", False)
            stream_callback = state.get("stream_callback")
            cancel_check = state.get("cancel_check")

            if stream:
                assistant_text = ""
                for chunk in self._client.chat_stream(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                ):
                    if cancel_check and cancel_check():
                        break
                    assistant_text += chunk
                    if stream_callback:
                        stream_callback(assistant_text)
                return {"response": assistant_text.strip()}

            assistant_text = self._client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return {"response": assistant_text}

        builder = StateGraph(AgentState)
        builder.add_node("llm", call_llm)
        builder.set_entry_point("llm")
        builder.add_edge("llm", END)
        return builder.compile()

    def run(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> str:
        state: AgentState = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "stream_callback": stream_callback,
            "cancel_check": cancel_check,
        }
        result = self._graph.invoke(state)
        return str(result.get("response", ""))
