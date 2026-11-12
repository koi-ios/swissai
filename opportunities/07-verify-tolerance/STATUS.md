# STATUS: 07-verify-tolerance

Updated 12 Nov 2026.

## Done

- **Research.**
  - `research/engines.md` covers determinism modes and their costs, engine CI tolerances, the weight checksums in SGLang 0.5.20 and vLLM 0.33.0 (read at the tags), and serving incidents.
  - `research/literature.md` covers TOPLOC, DiFR, Log Probability Tracking, Model Equality Testing, flips/KL, quantization formats, fault injection and fingerprinting.
  - Unverified items are marked as such in both.
- **servekit (3 commits on `verify-statistical`, patches in `patches/`).** The patches apply with `git am` onto `main` @ cd8f0b8.
  - `verify --calibrate N`, `--mode auto|exact|statistical|legacy`, `--alpha`, `--calibration-from`, `--weights` (SGLang `/weights_checker`), and capture format 2, which is backward compatible.
  - `servekit.faults`: 11 fault kinds, with `inject`, `list`, `diff` and `validate`.
  - Tests: 183 pass, 47 of them new. The 2 `test_launch.py` stager errors also fail on `main` on macOS (no `O_DIRECT`).
- **The fault suite works on real safetensors files:** Qwen3-0.6B BF16, the official Qwen3-0.6B-FP8 (block scales) and a locally made MXFP4 variant. The rewritten files are read back by the `safetensors` reference reader. `validate` passes all 42 GPU-study faults against the real Qwen3-1.7B, Qwen3-1.7B-FP8 and gpt-oss-20b headers.
- **CPU validation (`results/cpu/`).** 381 runs, 147 faults, 180 held-out benign runs.
  - False alarms: 3/180 with 16 calibration runs, 2/132 with 32. Fixed tolerances: 180/180.
  - Detection: 63/63 of faults above 3× the benign noise. Exact mode: 147/147.
  - FP8's noise is 17× BF16's.
  - The run caught and fixed a miscalibrated first `max_z` rule.
- **End to end on CPU.**
  - `study_driver.py` against `cpu/server.py`: `results/e2e_cpu_final/`.
  - The `servekit verify` CLI: `results/e2e_cli/`.
  - The lab's own TP4/TP8 fixtures analysed: `results/fixtures_tp4_vs_tp8.txt`.
- **Modal.** `modal_app.py` and `MODAL.md`:
  - `python modal_app.py plan` passes.
  - Flags checked against the SGLang 0.5.20 wheel and vLLM 0.33.0 source.
  - Estimated ~$14.6 for four arms; budget-capped worst case $24.6.
- **Drafts.** `drafts/PR.md` and `WRITEUP.md`. Hand-over notes for siblings are in the shared notes folder.

## Not done / blocked

- **The GPU study has not run.** There is no Modal account here, and spending is the user's call. Every GPU minute and dollar in `MODAL.md` is an estimate, and the engine start times are guesses until the smoke test runs.
- **No real cross-TP or cross-hardware calibration.** The only real TP pair is the two Llama-70B fixtures; the CPU study emulates TP 2 and 4.
- **The `/weights_checker` blind spot** (MXFP4 experts moved off the module) is a prediction from the source; the gpt-oss arm tests it.
- **Statistical-mode false alarms run slightly above nominal** (~1.6% against 1%), from settings calibration rarely samples.
- **No two-sample mode** yet; it would raise power for faults below the noise floor.

## Next 3 actions

1. **Account A.** Run `modal run modal_app.py --mode smoke --model qwen3-1.7b` (~$0.5), then the `qwen3-1.7b` and `qwen3-1.7b-fp8` studies (~$6.6). Then `modal volume get verify-tolerance-results / results/modal/` and `python -I cpu/report.py results/modal/qwen3-1.7b`.
2. **Account B.** Run the gpt-oss-20b study (~$5.6). It answers two questions: does `--enable-deterministic-inference` make MXFP4 MoE reproduce bitwise under load, and does `/weights_checker` see zeroed experts?
3. **Send to Xiaozhe.** Send the PR (`drafts/PR.md` + `patches/`) with the GPU tables, and propose `--calibrate 32 --weights` on the e2e `MODE=baseline` step.
