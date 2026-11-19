from __future__ import annotations


def unwrap_text_envelope(value: object) -> object:
    if not isinstance(value, dict):
        return value
    text = value.get("text")
    if not isinstance(text, str) or not text.startswith("AIDT:"):
        return value

    header, separator, original = text.partition("\n")
    if (
        set(value) != {"text"}
        or not separator
        or header not in {"AIDT:1:detect", "AIDT:1:evidence"}
    ):
        raise ValueError("Invalid text envelope.")
    try:
        original.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("Invalid text envelope.") from None
    return {"text": original, "action": header.removeprefix("AIDT:1:")}
