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
# 1. The A/A pilot: baseline against itself, in its own block.
python3 harness/measure.py schedule --arms base,base2 --pairs 6 --seed 1
python3 harness/measure.py record --out results/runs.jsonl --block pilot --arm base --pair 1 \
    --metric total_s=... --phase weights_s=... --cache page=cold --engine-commit ... --exit-code 0

# 2. The comparison, in another block. Run each line of the schedule as a fresh process.
python3 harness/measure.py schedule --arms base,fix --pairs 6 --seed 2
python3 harness/measure.py record --out results/runs.jsonl --block main --arm fix --pair 1 --metric total_s=...

# 3. The verdict: DESCRIPTIVE, NO-PILOT, NOT DISTINGUISHABLE or SUPPORTED.
python3 harness/measure.py analyze results/runs.jsonl --metric total_s --block main \
    --base base --test fix --aa-block pilot --aa base2
```

`record` stores the digest of `HARNESS.sha256` with every run, so a number is tied to the harness that produced it. `analyze` refuses two records for the same block, arm and pair, and prints how many runs it left out and why.
