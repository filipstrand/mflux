import json


def _extract_iptc_tag_value(iptc_binary: bytes, tag_id: int) -> bytes | None:
    i = 0
    while i + 5 <= len(iptc_binary):
        if iptc_binary[i] == 0x1C and iptc_binary[i + 1] == 0x02:
            this_tag = iptc_binary[i + 2]
            length = int.from_bytes(iptc_binary[i + 3 : i + 5], "big")
            start = i + 5
            end = start + length
            if end > len(iptc_binary):
                return None
            if this_tag == tag_id:
                return iptc_binary[start:end]
            i = end
        else:
            i += 1
    return None


def test_build_iptc_binary_long_json_prompt_does_not_warn(caplog):
    from mflux.utils.metadata_builder import MetadataBuilder

    # Build a prompt that is valid JSON and > 2000 bytes when UTF-8 encoded
    long_json_prompt = json.dumps({"prompt": "x" * 5000}, ensure_ascii=False)
    assert len(long_json_prompt.encode("utf-8")) > MetadataBuilder._IPTC_PROMPT_MAX_BYTES  # sanity

    with caplog.at_level("WARNING"):
        iptc_binary = MetadataBuilder.build_iptc_binary({"prompt": long_json_prompt})

    # No WARNING log should be emitted for structured JSON prompts.
    assert "Prompt is too long" not in caplog.text

    caption = _extract_iptc_tag_value(iptc_binary, 120)  # 2:120 Caption/Abstract
    assert caption is not None
    assert len(caption) <= MetadataBuilder._IPTC_PROMPT_MAX_BYTES

    decoded = caption.decode("utf-8", errors="replace")
    assert "Structured JSON prompt" in decoded
