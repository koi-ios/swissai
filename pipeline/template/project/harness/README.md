# Harness

The harness is whatever turns a run into a number: launch scripts, the run recorder, the correctness oracle and the analysis. Once frozen, it is read-only to any session that changes the system under test.

## Freeze

A harness may be frozen only when all four hold (D1, stage 3):

1. A smoke run passed on the real target (the real image, a real GPU), not only against a mock.
2. The baseline reproduces a number someone else measured, within the noise.
3. An A/A pilot (baseline against itself, 6 interleaved pairs) gave the noise floor.
4. The correctness oracle passes on the baseline and fails on the known-bad patch.

Then record the hash and commit it:

```bash
find harness -type f ! -name '*.pyc' | LC_ALL=C sort | xargs shasum -a 256 > HARNESS.sha256
```

`shasum -a 256 -c HARNESS.sha256` must pass before every sweep and in `VERIFY.md`. To change a frozen harness, write down why, re-run the baseline and the pilot, and record a new hash.

## Files

- `measure.py`: run records, interleaved schedules, and the small-n analysis. Copied from the template; do not edit it here.
- `known_bad/`: a change that must make the oracle fail. If the harness accepts it, the harness is broken.
- Project-specific launch and oracle scripts go beside these.

## Run order for a comparison

```bash
python3 harness/measure.py schedule --arms base,fix --pairs 6 --seed 1    # prints the order
# run each line as a fresh process; record it:
python3 harness/measure.py record --out results/runs.jsonl --arm fix --pair 1 --metric total_s=...
python3 harness/measure.py analyze results/runs.jsonl --metric total_s --base base --test fix
```
