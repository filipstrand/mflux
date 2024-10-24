import mlx.optimizers as optim


class Optimizer:
    @staticmethod
    def setup_optimizer():
        learning_rate = 1e-4
        warmup_steps = 1
        iterations = 500
        grad_accumulate = 1
        warmup = optim.linear_schedule(0, learning_rate, warmup_steps)
        cosine = optim.cosine_decay(learning_rate, iterations // grad_accumulate)
        lr_schedule = optim.join_schedules([warmup, cosine], [warmup_steps])
        optimizer = optim.Adam(learning_rate=lr_schedule)
        return optimizer
