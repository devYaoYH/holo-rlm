# Paired hotel free-generation pilot

## Protocol

- Checkpoints: Holo-3.1-4B and its Qwen3.5-4B base, run separately on one RTX 5090.
- Harness: repository `holo-capture run --backend local` with the HoloDesktop runtime 0.1.10 tool contract, normalized 0–1000 coordinates, and the repository's cheapest-hotel system prompt.
- Generation: independent free generation, temperature 0.8, at most four model actions, stop on the first click.
- Visual memory: the current native screenshot plus at most two retained screenshots.
- Cases: frozen items from `benchmarks/frozen_eval_v2.json`, spanning three currencies, six- and seven-hotel lists, and runner-up gaps from 2 to 69.

## Golden-route audit

The primary cohort contains six items whose deterministic reference route needs no more than three moderate scroll actions before the click:

`test-0005`, `test-0010`, `test-0020`, `test-0026`, `test-0040`, and `test-0070`.

`test-0000` was collected but excluded from the primary cohort after audit: its cheapest hotel is at the top, and inspecting the bottom then returning to a visible click target requires four scroll actions. In this fixture, no top-placement case satisfies the hard three-scroll route bound. The retained diagnostic remains available in the archive.

## Free-generation results

| Checkpoint | Success | Invalid structured action | Step limit | Incorrect click |
|---|---:|---:|---:|---:|
| Holo-3.1-4B | 2/6 | 4/6 | 0/6 | 0/6 |
| Qwen3.5-4B | 0/6 | 2/6 | 2/6 | 2/6 |

Holo completes `test-0010` and `test-0020`. Qwen shows some correct price retrieval in its notes but either fails the action interface, scrolls past the four-action budget, or clicks the wrong hotel. The small pilot therefore supports action-policy and structured-output fragility, but it does not cleanly isolate visual understanding from stopping behavior or formatting.

## Activation traces

- Holo `test-0010`: a traced replicate succeeds with two scrolls and the correct click. Three native-resolution traces capture all three decisions.
- Qwen `test-0010`: one native-resolution trace is captured before the first response fails to emit a valid structured action.
- The Holo final-decision value-norm rollout spans all three retained frames. These new maps are raw descriptive attribution; they do not yet include the same-image diverse-instruction subtraction used elsewhere in the deck.

The checksummed raw archive is stored locally under `data/remote-results/hotel-freegen-paired-v1-20260922/`. Reproducible aggregate results are in `artifacts/screenspot-presentation/hotel-freegen-paired-v1-summary.json`; generated viewers and static previews are under `artifacts/screenspot-presentation/hotel-freegen-paired-v1-attribution/`.
