import enum


class NoiseSchedulerType(enum.Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    SQRT = "sqrt"
    SCALED_LINEAR = "scaled_linear"

    @classmethod
    def choices(cls) -> list[str]:
        """Return a list of all available noise scheduler values as strings."""
        return [e.value for e in cls]

    @classmethod
    def from_string(cls, value: str) -> "NoiseSchedulerType":
        """Convert a string to the corresponding enum member."""
        for member in cls:
            if member.value == value:
                return member
        valid_values = ", ".join(cls.choices())
        raise ValueError(f"Invalid noise scheduler type: {value}. Valid choices: {valid_values}")
