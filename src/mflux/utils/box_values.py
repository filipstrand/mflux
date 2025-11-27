from dataclasses import dataclass


@dataclass
class AbsoluteBoxValues:
    top: int
    right: int
    bottom: int
    left: int


class BoxValueError(ValueError):
    pass


@dataclass
class BoxValues:
    top: int | str
    right: int | str
    bottom: int | str
    left: int | str

    def normalize_to_dimensions(self, width, height) -> AbsoluteBoxValues:
        parts = []
        dimension_base = [height, width, height, width]
        for index, part in enumerate([self.top, self.right, self.bottom, self.left]):
            if isinstance(part, str) and part.endswith("%"):
                parts.append(int(int(part.strip("%")) / 100 * dimension_base[index]))
            else:
                # simple integer value
                parts.append(int(part))
        return AbsoluteBoxValues(*parts)

    @staticmethod
    def parse(value, delimiter=",") -> "BoxValues":
        parts = []
        for part_value in value.strip().split(delimiter):
            try:
                part = int(part_value.strip())
                parts.append(part)
            except ValueError:  # noqa: PERF203
                # not an int - is it a %?
                if (part_value := part_value.strip()).endswith("%"):
                    parts.append(part_value)
                else:
                    raise BoxValueError(f"Invalid padding value: {part_value}")

        if len(parts) == 1:
            # If only one value is provided, apply to all sides
            return BoxValues(top=parts[0], right=parts[0], bottom=parts[0], left=parts[0])
        elif len(parts) == 2:
            # If two values: first is top/bottom, second is left/right
            return BoxValues(top=parts[0], right=parts[1], bottom=parts[0], left=parts[1])
        elif len(parts) == 3:
            # If three values: top, left/right, bottom
            return BoxValues(top=parts[0], right=parts[1], bottom=parts[2], left=parts[1])
        elif len(parts) == 4:
            # If four values: top, right, bottom, left
            return BoxValues(top=parts[0], right=parts[1], bottom=parts[2], left=parts[3])
        else:
            raise BoxValueError(
                "Invalid outpaint padding box value format: {value} "
                "Expected: 1 (all-sides), 2 (top/bottom, left/right) "
                "or 4 (top, right, bottom, left)values of int or percentages. e.g. 10px, 20%"
            )
