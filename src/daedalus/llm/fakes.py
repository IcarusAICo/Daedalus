"""Test doubles for the LLM gateway. Importable from production code so the
Planner / Implementor / Learner can be exercised without hitting Bedrock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from daedalus.llm.gateway import LLMCall, LLMResponse


@dataclass
class FakeGateway:
    """Records calls and replies with a scripted response.

    Two ways to script:

    - ``responses``: a fixed list, popped in order.
    - ``responder``: a callable invoked with each :class:`LLMCall`, returning
      the response string. Wins over ``responses`` when both are set.
    """

    responses: list[str] = field(default_factory=list)
    responder: Callable[[LLMCall], str] | None = None
    calls: list[LLMCall] = field(default_factory=list)
    tracer: object | None = None

    def set_tracer(self, tracer: object | None) -> None:
        self.tracer = tracer

    def complete(self, call: LLMCall) -> LLMResponse:
        self.calls.append(call)
        if self.responder is not None:
            text = self.responder(call)
        elif self.responses:
            text = self.responses.pop(0)
        else:
            raise AssertionError("FakeGateway has no scripted response left")
        model = "fake/model"
        return LLMResponse(
            role=call.role,
            model=model,
            content=text,
            raw={"choices": [{"message": {"content": text}}]},
            duration_s=0.0,
            usage={},
        )
