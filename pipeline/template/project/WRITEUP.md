# Title: what was found or built, in one line

*Date. Two pages at most. Labels: **MEASURED** (n, setup, date), **PREDICTED** (model and its validation error), **ESTIMATE** (basis). Nothing here is a measurement unless it is labelled one.*

## 1. Problem, with numbers

Whose problem it is, where it shows (issue, log, trace), and how large it is. Cite the lab's own data first.

## 2. What exists, and what is new here

Prior work with credit: the papers, the upstream code, the people. Then, separately, what this adds. If the contribution is measurement or engineering, say so; do not call it research novelty.

## 3. What was built

The idea in three or four sentences, then the pieces, each with its path.

## 4. Results

| Claim | Value | Label | n | Setup (hardware, versions, cache state) | Script | Raw records |
|---|---|---|---|---|---|---|
| | | | | | | |

Report medians with every raw value, the ratio of medians, the A/A range and, from 6 pairs up, the sign-test p. Give absolute numbers beside ratios. Show where the change makes things worse. State the baseline: current upstream at a named commit.

## 5. How to run it

```bash
# smoke test, a few minutes, no GPU

# every number in section 4, with expected time and cost
```

## 6. What it would take on the cluster, and the risks

## 7. How AI was used

One specific paragraph: which tool drafted what, what I designed, what I reviewed line by line, and which numbers I re-derived myself.

## 8. Pitch

Three lines for the person this is for.
