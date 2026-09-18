.PHONY: doctor smoke demo benchmark holo test lint validate clean-fixture

BACKEND ?= scripted
SEED ?= 0
VARIANT ?= 0

doctor:
	uv run holo-capture doctor --backend $(BACKEND)

smoke:
	uv run holo-capture smoke --backend $(BACKEND)

demo:
	uv run holo-capture run --backend $(BACKEND) --seed $(SEED) --variant $(VARIANT)

benchmark:
	uv run holo-capture benchmark --backend $(BACKEND) --count 20

holo:
	uv run holo-capture holo --backend $(BACKEND) --seed $(SEED)

test:
	uv run pytest

lint:
	uv run ruff check .

validate:
	uv run holo-capture validate data/trajectories/v0 --allow-runtime-only

clean-fixture:
	uv run holo-capture reset --seed $(SEED) --variant $(VARIANT)
