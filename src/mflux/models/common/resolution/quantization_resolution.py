import logging

from mflux.models.common.resolution.actions import QuantizationAction, Rule

logger = logging.getLogger(__name__)


class QuantizationResolution:
    RULES = frozenset(
        {
            Rule(priority=0, name="none", check="none_none", action=QuantizationAction.NONE),
            Rule(priority=1, name="on_the_fly", check="none_any", action=QuantizationAction.REQUESTED),
            Rule(priority=2, name="pre_quantized", check="any_none", action=QuantizationAction.STORED),
            Rule(priority=3, name="conflict", check="any_any", action=QuantizationAction.STORED),
        }
    )

    @staticmethod
    def resolve(stored: int | None, requested: int | None) -> tuple[int | None, str | None]:
        for rule in sorted(QuantizationResolution.RULES, key=lambda r: r.priority):
            if QuantizationResolution._check(rule.check, stored, requested):
                logger.debug(
                    f"Quantization resolution: stored={stored}, requested={requested} "
                    f"→ rule '{rule.name}' ({rule.action.value})"
                )
                return QuantizationResolution._execute(rule, stored, requested)

        raise ValueError(f"Unexpected quantization state: stored={stored}, requested={requested}")

    @staticmethod
    def _check(check: str, stored: int | None, requested: int | None) -> bool:
        if check == "none_none":
            return stored is None and requested is None
        if check == "none_any":
            return stored is None and requested is not None
        if check == "any_none":
            return stored is not None and requested is None
        if check == "any_any":
            return stored is not None and requested is not None
        return False

    @staticmethod
    def _execute(rule: Rule, stored: int | None, requested: int | None) -> tuple[int | None, str | None]:
        if rule.action == QuantizationAction.NONE:
            return None, None
        if rule.action == QuantizationAction.REQUESTED:
            return requested, None
        if rule.action == QuantizationAction.STORED:
            warn = rule.name == "conflict" and stored != requested
            warning = f"Model is pre-quantized at {stored}-bit. Ignoring -q {requested} flag." if warn else None
            return stored, warning
        return None, None
