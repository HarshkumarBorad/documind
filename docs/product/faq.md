# Aurora — Frequently Asked Questions

## General

**Q: What is Aurora?**
A: Aurora is a managed compute and storage platform for ML and data engineering workloads. We provide on-demand GPU and CPU compute, an S3-compatible object store, and a CLI/SDK/Desktop client to manage it all.

**Q: Which regions are available?**
A: We currently operate in Frankfurt (`eu-central-1`), Amsterdam (`eu-west-1`), and Northern Virginia (`us-east-1`). New regions are added based on customer demand — let us know via support if you need one we don't list.

**Q: How is pricing calculated?**
A: Compute is billed per second of usage at the instance's hourly rate. Storage is billed at the end of each month based on total GB-hours stored. Egress is included in the storage cost up to 10× your stored volume per month.

## Compute

**Q: What instance types are available?**
A: CPU-only (c-series), GPU instances with NVIDIA A100 (`gpu-a100`) and H100 (`gpu-h100`), and memory-optimized (m-series). See the pricing page for the current list.

**Q: How long can a job run?**
A: There is no hard limit. We have customers running training jobs for over a month continuously. For very long jobs we recommend checkpointing — see the SDK Quickstart for the checkpointing API.

**Q: What happens if my job runs out of memory?**
A: The job is killed and marked as `oom_killed`. Logs up to the kill point are preserved. You can resubmit with a larger instance type.

## Storage

**Q: Is my data encrypted at rest?**
A: Yes, all objects are encrypted with AES-256 by default. Customer-managed encryption keys (KMS) are available on the Enterprise tier.

**Q: Can I use S3-compatible tools?**
A: Yes. Point any S3 client at `https://s3.aurora.example` with your Aurora credentials. We support the S3 API up to the 2024 version.

## Connectivity & troubleshooting

**Q: How do I troubleshoot connection errors?**
A: Check your firewall and ensure HTTPS access to `api.aurora.example:443`. Corporate proxies may need configuration in **Settings → Network**. If the issue persists, check the service status page at `status.aurora.example`.

## Support

**Q: How do I contact support?**
A: Free tier — community forum at `forum.aurora.example`. Pro tier — email `support@aurora.example`, response within 24 hours. Enterprise tier — dedicated Slack channel and a 1-hour response SLA.

**Q: Where can I see service status?**
A: Live status page at `status.aurora.example`. We post incident updates there in real time.
