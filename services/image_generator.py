from openai import AsyncOpenAI
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_client():
    return AsyncOpenAI()


async def generate_image(prompt: str) -> str:
    client = _get_client()

    response = await client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        n=1,
        size="1024x1024",
        quality="low",
    )

    return response.data[0].b64_json
