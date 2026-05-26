# Aurora Python SDK — Quickstart

The Aurora Python SDK provides a typed, idiomatic Python interface to the Aurora API.

## Requirements

- Python **3.10 or higher**.
- An Aurora API key (see the API Reference for how to obtain one).

## Installation

```bash
pip install aurora-sdk
```

## Authentication

The SDK reads your API key from the `AURORA_API_KEY` environment variable by default:

```bash
export AURORA_API_KEY=sk_your_key_here
```

You can also pass it explicitly when constructing the client:

```python
from aurora_sdk import AuroraClient

client = AuroraClient(api_key="sk_your_key_here")
```

## Submitting a compute job

```python
from aurora_sdk import AuroraClient

client = AuroraClient()

job = client.compute.submit(
    image="myregistry/myimage:latest",
    command=["python", "train.py"],
    resources={"cpu": 4, "memory_gb": 16, "gpu": "a100:1"},
)
print(f"Submitted job {job.id}")

# Poll until done. Raises if the job times out.
final = job.wait(timeout=3600)
print(final.status, final.exit_code)
```

## Storage operations

```python
client.storage.upload("my-bucket", "model.bin", local_path="./model.bin")
client.storage.download("my-bucket", "model.bin", local_path="./downloaded.bin")
```

## Async support

The SDK ships an async client for use in asyncio applications:

```python
from aurora_sdk import AsyncAuroraClient

async def main():
    async with AsyncAuroraClient() as client:
        job = await client.compute.submit(
            image="myregistry/myimage:latest",
            command=["python", "train.py"],
            resources={"cpu": 4, "memory_gb": 16, "gpu": "a100:1"},
        )
        result = await job.wait(timeout=3600)
```

## Error handling

All SDK errors inherit from `aurora_sdk.AuroraError`. Common subclasses:

- `AuthenticationError` — invalid or revoked API key.
- `RateLimitError` — exceeded your tier's request rate.
- `NotFoundError` — referenced resource (job, object) does not exist.
- `ValidationError` — request body failed server-side validation.

The async client raises the same exception types as the sync client.
