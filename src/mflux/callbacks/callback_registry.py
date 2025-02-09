from mflux.callbacks.callback import BeforeLoopCallback, InLoopCallback, InterruptCallback


class CallbackRegistry:
    in_loop = []
    before_loop = []
    interrupt = []

    @staticmethod
    def register_in_loop(callback: InLoopCallback) -> None:
        CallbackRegistry.in_loop.append(callback)

    @staticmethod
    def register_before_loop(callback: BeforeLoopCallback) -> None:
        CallbackRegistry.before_loop.append(callback)

    @staticmethod
    def register_interrupt(callback: InterruptCallback) -> None:
        CallbackRegistry.interrupt.append(callback)

    @staticmethod
    def in_loop_callbacks() -> list[InLoopCallback]:
        return CallbackRegistry.in_loop

    @staticmethod
    def before_loop_callbacks() -> list[BeforeLoopCallback]:
        return CallbackRegistry.before_loop

    @staticmethod
    def interrupt_callbacks() -> list[InterruptCallback]:
        return CallbackRegistry.interrupt
