import json


def _flatten(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for v in value for item in _flatten(v)]
    if isinstance(value, dict):
        return [item for v in value.values() for item in _flatten(v)]
    return [str(value)]


async def build_image_prompt(context: dict) -> str:
    parts = _flatten(context)
    combined = ", ".join(p.strip() for p in parts if p.strip())

    return (
        f"Professional food photography, photorealistic, studio lighting, "
        f"shallow depth of field, high detail, appetizing: {combined}. "
        f"Shot on high-end camera, 85mm lens, commercial food photography style."
    )
