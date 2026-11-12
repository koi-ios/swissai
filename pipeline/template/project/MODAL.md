# Modal runbook: NN-slug

Agents write this file and dry-run the app. A person launches every paid run (D10).

## Before the first paid run

- [ ] One workspace, with a spend limit set in the dashboard
- [ ] `python3 modal_app.py --dry-run` passes locally and prints the planned cost
- [ ] Every function sets `timeout=` and `retries=0`
- [ ] The GPU type is pinned, for example `gpu="H100!"` or `gpu="A100-80GB"`; plain `H100` may run on other hardware
- [ ] The app records one line per run with `harness/measure.py record`, including the GPU UUID and the cache states
- [ ] `make budget PROJECT=NN COST=<usd>` accepts the planned cost

## Plan

Cheapest decisive step first. Stop rules are written before the run.

| Step | Command | GPU (pinned) | Runs | Est. minutes | Est. cost | Decides | Stop rule |
|---|---|---|---|---|---|---|---|
| smoke | `modal run modal_app.py::smoke` | | 1 | | | the app works on the real image | fails twice: stop and fix locally |
| A/A pilot (`--block pilot`, arms `base,base2`) | | | 6 pairs | | | the noise floor | CoV above 10%: find the noise source first |
| main (`--block main`, arms `base,<change>`) | | | 6 pairs | | | the claim | `analyze` says SUPPORTED or NOT DISTINGUISHABLE: stop |
| escalation, once (`--block main10`) | | | 10 pairs | | | an effect within twice the A/A range | still not distinguishable: report that and stop |

Each step records into its own block, so pair numbers never collide. A rerun gets a new block name; a failed run is recorded with its exit code, not deleted.

Estimated cost per run = container seconds x (GPU rate + cores x CPU rate + GiB x memory rate). Take the rates from the provider's pricing page on the day and note the date.

## After each run

- Add a line to `budget/ledger.csv` with `make spend` (planned and actual cost, the app ID, the date).
- Fetch the results: `modal volume get <volume> / results/modal/`, then `python3 harness/measure.py analyze ...`.
- A failed run goes to `results/quarantine/` with its log. It is not retried automatically.

## Limits to state in the write-up

Modal has no Lustre. Containers are not VMs: `/dev/shm` is limited and the page cache cannot be dropped. A number measured here is a Modal number.
