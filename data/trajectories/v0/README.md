# Trajectory v0 sample dataset

This directory contains the checked-in, synthetic v0 trajectory fixtures used
to exercise replay, validation, and research metadata handling. The fixtures
run against the local booking demo and use the deterministic scripted backend;
they do not contain model weights, credentials, or external service data.

Runtime-generated local-model trajectories and activation traces remain ignored
under `data/` because they may contain machine-specific screenshots and model
outputs. Generate those locally with `holo-capture run --backend local` when
needed.
