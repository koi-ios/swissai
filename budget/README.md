# Budget

Decision D10: one capped workspace, pinned hardware, the user launches.

- `plan.csv`: the Modal steps planned on 12 Nov, in the order to run them (cheapest decisive step first). Estimates, not measurements. The rows sum to about $75; `opportunities/README.md` says "about $88 if everything runs", which counts steps the table does not list.
- `ledger.csv`: what was actually spent, one row per run. Append with `make spend`; never edit a row.
- Caps live in `pipeline/workspace.toml`: $25 a project, $30 a month (one workspace's free credit), $90 in all.

Before a paid run:

1. `make budget PROJECT=02 COST=6` must say "allowed".
2. The spend limit is set in the provider's dashboard. That is the second stop; the gate above is the first.
3. The app pins its GPU type (`H100!`, not `H100`), sets `retries=0` and a timeout on every function.

After it: `make spend PROJECT=02 STEP="gpu_check + experiment" ACTUAL=5.40 GPU="H100!" SECONDS=2400 PLANNED=6 APP=ap-...`, then `make portfolio`.

Open, and the user's to settle (see `pipeline/MIGRATION.md`, section 7): the plan of 12 Nov spread these steps over three accounts with $30 each (last column). The provider's terms have no clause on several accounts and bar third-party use; the credit is monthly. Until that is settled, plan on one workspace and at most $30 a month.
