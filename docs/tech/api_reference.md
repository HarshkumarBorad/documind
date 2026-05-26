# Aurora API Reference

The Aurora REST API exposes our compute and storage primitives over HTTPS. All endpoints are served at `https://api.aurora.example/v1`.

## Authentication

All requests require a Bearer token in the `Authorization` header:

```
Authorization: Bearer <your_api_key>
```

API keys are issued from the Aurora Console under **Account → API Keys**. Each key is scoped to a specific project and can be rotated or revoked at any time.

## Rate limits

| Tier | Per-minute limit | Daily cap |
|---|---|---|
| Free | 60 requests / minute | 10,000 requests / day |
| Pro | 600 requests / minute | none |
| Enterprise | custom | custom |

Exceeding the limit returns `429 Too Many Requests` with a `Retry-After` header indicating how many seconds to wait before retrying.

## Compute endpoints

### `POST /compute/jobs`
Submit a new compute job.

Request body:

```json
{
  "image": "myregistry/myimage:latest",
  "command": ["python", "train.py"],
  "resources": {"cpu": 4, "memory_gb": 16, "gpu": "a100:1"}
}
```

Returns `201 Created` with the job ID in the body and `Location` header.

### `GET /compute/jobs/{id}`
Fetch job status. Status values: `pending`, `running`, `succeeded`, `failed`, `cancelled`, `oom_killed`.

### `DELETE /compute/jobs/{id}`
Cancel a running or pending job. No-op if the job is already in a terminal state.

## Storage endpoints

### `PUT /storage/objects/{bucket}/{key}`
Upload an object. Supports multipart upload for files >100MB via the `Content-Type: multipart/form-data` header.

### `GET /storage/objects/{bucket}/{key}`
Download an object. Supports `Range` requests for partial reads.

### `DELETE /storage/objects/{bucket}/{key}`
Delete an object. Soft-deleted by default; pass `?hard=true` to bypass the 7-day recovery window.

## SDK

Official SDKs are available for **Python**, **TypeScript**, and **Go**. See the SDK Quickstart guide for installation and usage.

## Versioning

The API follows semantic versioning. Breaking changes are released in a new major version (e.g. `/v2`); minor versions add fields without removing existing ones. Deprecated endpoints are supported for at least 12 months after their replacement is announced.
