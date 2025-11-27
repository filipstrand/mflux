from __future__ import annotations

from typing import TYPE_CHECKING

from mflux.callbacks.callback import AfterLoopCallback, BeforeLoopCallback, InLoopCallback, InterruptCallback

if TYPE_CHECKING:
    from mflux.callbacks.generation_context import GenerationContext
    from mflux.models.common.config.config import Config


class CallbackRegistry:
    def __init__(self):
        self.in_loop: list[InLoopCallback] = []
        self.before_loop: list[BeforeLoopCallback] = []
        self.interrupt: list[InterruptCallback] = []
        self.after_loop: list[AfterLoopCallback] = []

    def register(self, callback) -> None:
        if hasattr(callback, "call_before_loop"):
            self.before_loop.append(callback)
        if hasattr(callback, "call_in_loop"):
            self.in_loop.append(callback)
        if hasattr(callback, "call_after_loop"):
            self.after_loop.append(callback)
        if hasattr(callback, "call_interrupt"):
            self.interrupt.append(callback)

    def start(self, seed: int, prompt: str, config: Config) -> GenerationContext:
        from mflux.callbacks.generation_context import GenerationContext

        return GenerationContext(self, seed, prompt, config)

    def before_loop_callbacks(self) -> list[BeforeLoopCallback]:
        return self.before_loop

    def in_loop_callbacks(self) -> list[InLoopCallback]:
        return self.in_loop

    def after_loop_callbacks(self) -> list[AfterLoopCallback]:
        return self.after_loop

    def interrupt_callbacks(self) -> list[InterruptCallback]:
        return self.interrupt
