class CfgEnd:
    """CFG truncation (--cfg-end): guidance runs on the first steps only.

    Guidance shapes the result in the early steps (composition); the late steps mostly refine.
    ``cfg_end`` is the fraction of steps that run CFG, rounded to whole steps, and at least the
    first step does. The remaining steps run the conditional pass alone, as at guidance 1.0, so
    they skip the unconditional pass. Ideogram 4 introduced it (#441).

    Usage in a model's denoise loop::

        cfg_cutoff = CfgEnd.cutoff(config.num_inference_steps, cfg_end)
        for t in config.time_steps:
            negative = negative_embeds if t < cfg_cutoff else None
            noise = ...  # the conditional pass, and the unconditional one only when negative is not None
    """

    @staticmethod
    def cutoff(num_steps: int, cfg_end: float | None) -> int:
        """The steps before this index run CFG: all of them when ``cfg_end`` is None."""
        if cfg_end is None:
            return num_steps
        if not 0 < cfg_end <= 1:
            raise ValueError(f"cfg_end must be > 0 and <= 1, got {cfg_end}")
        return max(1, int(round(float(cfg_end) * num_steps)))

    @staticmethod
    def generation_parameters(cfg_end: float | None) -> dict:
        """The steps without guidance change the image, so a run that set cfg_end records it.

        Every other image's metadata is unchanged; --config-from-conf reads it back.
        """
        return {"cfg_end": cfg_end} if cfg_end is not None else {}
