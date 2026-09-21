# Remote execution checklist

Follow the detailed repository guide at `docs/remote-gpu-experiments.md`. Use this checklist while operating an SSH host.

## Before connecting

- Confirm the user supplied the host, username, authentication method, and intended remote repository/output paths.
- Never place private keys or provider tokens in the repository, shell history, logs, manifests, or result archives.
- Treat instance creation, shutdown, deletion, and paid-storage changes as separate external actions requiring explicit user authorization.

## On the host

1. Print the repository commit and concise dirty status.
2. Run `nvidia-smi` and `holo-gpu-doctor --require-cuda`.
3. Verify the checkpoint and benchmark data paths without listing secrets.
4. Run both test suites before inference.
5. Start the server on `127.0.0.1`; retain its log.
6. Select and record the server profile: native-resolution/non-eager for ScreenSpot-Pro, or eager attention with an explicit per-frame pixel budget for attribution.
7. Run a one-item or one-case smoke experiment.
8. Inspect its response, trace ID, token logprob artifact, processor pixel limit, and free disk space.
9. Launch the requested resumable run. Use distinct outputs for distinct shards.
10. Monitor item counts, errors, GPU memory, and disk space. Resume rather than replacing partial results.

## Retrieval

Package with `holo-capture package-results`, retrieve both archive and checksum, and verify locally. Record the SHA-256 digest in the handoff. Leave the remote copy intact until the user confirms retention or deletion.

## Failure handling

- If the server exits during model load, preserve logs and report peak GPU/host memory; do not change dtype or quantize without approval.
- If an item fails, preserve its atomic `result.json`; use resume for the remaining items.
- If disk space is low, stop starting new items, package completed outputs if possible, and ask before deleting traces.
- If clean/corrupt activation alignment fails, fix the experimental pair or define a reviewed alignment method; never coerce tensor positions.
