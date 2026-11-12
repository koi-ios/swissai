# Evidence listings: swiss-ai/model-launch (origin/main 02b74ce, gathered 12 Nov 2026)

## Remote branches (git branch -r, 53 incl. origin/HEAD pointer line)
```
origin
origin/AA/bump-dependencies
origin/AA/inf_vllm
origin/AA/move-images-out
origin/AA/muse-glimmer
origin/AA/nvidia-image-names
origin/AA/vllm-new-image
origin/AryanXiaozhe/Scaling
origin/ci/async-image-builds
origin/ci/cancel-stale-job
origin/ci/catalog-only-envs-images
origin/ci/never-auto-rebuild-release-images
origin/ci/scan-allowlist-package-tests
origin/ci/transfer-methods
origin/claude/claude-md-mmej3mzjxx4x62fi-Luroy
origin/factory/deepseek-ai__deepseek-v4.1-flash-1017-1d55bf2d
origin/factory/huggingfacetb__smollm2-135m-instruct-537f0b17
origin/factory/huggingfacetb__smollm2-135m-instruct-6d321ba9
origin/factory/huggingfacetb__smollm2-135m-instruct-c52187a8
origin/factory/huggingfacetb__smollm2-135m-instruct-f695f533
origin/factory/huggingfacetb__smollm2-135m-instruct-ffd73be6
origin/factory/meta-models__muse-spark-open-816b-a42b-nvfp4-0d6e2e14
origin/factory/meta-models__muse-spark-open-816b-a42b-nvfp4-c3de9a9b
origin/factory/moonshotai__kimi-k3.1-pro-be315b30
origin/factory/qwen__qwen2.5-0.5b-instruct-e153ea5d
origin/factory/qwen__qwen3.8-27b-a18b4e4c
origin/factory/xiaomimimo__mimo-v2.6-flash-rl-5c05920f
origin/feat/add-qwen3.5-27b
origin/feat/apertus2-chonk-example
origin/feat/firecrest-launcher-qos
origin/feat/jsc-jupiter
origin/feat/model-authorization
origin/feat/model-authorization-rebased
origin/feat/multi-cluster-endpoints
origin/feat/peer-labels
origin/feat/shadow-of
origin/fix/build-image-userinfo-group
origin/fix/dev-bootstrap-peer-id
origin/fix/hadolint-py-wheel-corruption
origin/fix/vllm-apertus-1.5-align-entrypoint
origin/images/arch-aware-cuda13-and-sglang-0.5.20
origin/images/arch-aware-vllm-cuda13
origin/infomaniak-apertus
origin/loleg-readme-1
origin/main
origin/opentela-per-node-identity
origin/pr-170
origin/robmsmt/apertus1.5-launch-scripts
origin/sglang_apertus
origin/tool-call-fix
origin/upgrade-apertus
origin/vllm-cxi-slingshot-bench
origin/vllm-upgrade
```

## Last 100 commits on origin/main: hash|date|author|parents|subject
```
02b74ce|2026-11-04|Rob|examples: add deepseek-ai/DeepSeek-V4.1-Flash-1017 (vllm) (#268)
6f8bea0|2026-11-04|Rob|examples: add XiaomiMiMo/MiMo-V2.6-Flash-RL (+ image vllm_nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423) (#231)
898b358|2026-11-04|Rob|examples: add HuggingFaceTB/SmolLM2-135M-Instruct (vllm) (#260)
72b8c1c|2026-11-04|Rob|ci: make lightweight integration tests opt-in (#282)
001a3ac|2026-11-04|Faruk Zahiragić|Add AGENTS.md requiring a dedicated branch per change (#278)
9965ba6|2026-11-04|Rob|Run sml on KISTI: stock pyxis container runtime, extra sbatch args, working dir (#280)
dc10d59|2026-10-29|Rob|fix(launchers): give the dev OpenTela bootstrap its own peer ID (#274)
521da02|2026-10-19|Aryan Ahadinia|envs: migrate tested SGLang examples to sglang_base, drop Kimi-K2 and sglang_0.5.20 (#271)
74da8ab|2026-10-19|Aryan Ahadinia|Migrate vLLM examples to vllm_base (#265)
c6c0e87|2026-10-18|Aryan Ahadinia|Stop building container images in model-launch (#259)
405e54d|2026-10-18|Aryan Ahadinia|cli: pass a flag-only --served-model-name to the framework (#269)
5d1f50b|2026-10-07|Mel Zuercher|launchers: configurable working directory (default stays ~/.sml) (#261)
1b3f599|2026-09-28|Rob|examples: add Qwen/Qwen3.8-27B (#246)
6f1ba2a|2026-09-28|Rob|examples: add zai-org/GLM-5.3 (vllm) (#240)
bcfe0e8|2026-09-28|Rob|examples: add XiaomiMiMo/MiMo-V2.6-Pro-RL (vllm) (#239)
1c12244|2026-09-28|Rob|examples: add zai-org/GLM-5.3-Flash (vllm) (#230)
1c0c7c4|2026-09-28|Rob|examples: add deepseek-ai/DeepSeek-V4.1-Flash (vllm) (#229)
d9dd73e|2026-09-26|Rob|ci: a release build must not die on chmod of the shared release directory (#258)
9ce8ffb|2026-09-25|Rob|images: arch-aware vllm_cuda13 template (split from #235) (#257)
970a965|2026-09-25|Rob|images: add sglang_0.5.20 (sglang 0.5.20) (#245)
1256c9d|2026-09-25|Rob|images: add vllm_0.30.0 (vllm v0.30.0) (#234)
180c8d2|2026-09-25|Rob|ci: manifest gate reads scan checks with GET; a tick does not rescan what already passed (#256)
586aa7d|2026-09-25|Rob|ci: allow PyCryptodome's self-test keys; one image's scan no longer blocks another's manifest (#255)
871f862|2026-09-25|Rob|ci: the finisher completes a check from the registry when the image is already there (#254)
719e9f3|2026-09-25|Rob|ci: a cluster job pushes each finished build to GHCR with a token the finisher lends it (#253)
b14abe0|2026-09-25|Rob|ci: the finisher rides out one cluster's FirecREST being down or unable to stage a download (#252)
403a7b9|2026-09-25|Rob|ci: harden the image build pipeline (#250)
ff8d4e9|2026-09-24|Rob|ci: the runner pushes the build's OCI archive to GHCR; the App only dispatches (#249)
6836925|2026-09-24|Rob|ci: build auth as a GitHub App; remove stale context files before upload; OCI source label (#248)
b7d63e5|2026-09-24|Rob|ci: image builds submit and exit; a scheduled workflow finishes, scans and merges them (#243)
ecfe37f|2026-09-24|Rob|ci: build_image.py adopts its SLURM job by name and rides out FirecREST errors (#237)
364e8c1|2026-09-24|Rob|examples: drop Apertus-1.5-8B sft-capfilter innovator-fix-it23409 — its checkpoint is gone from capstor (#238)
ae7497a|2026-09-24|Rob|catalog: add Qwen/Qwen3.8-27B (vllm, sglang) (#228)
d901a07|2026-09-24|Rob|ci: catalog-only PRs may add env TOMLs and image Dockerfiles (#233)
05c4854|2026-09-24|Rob|ci: build_image.py reads the primary group like the launcher (FirecREST 2.6.0+) (#232)
3df4362|2026-09-21|Rob|catalog: add Qwen/Qwen2.5-0.5B-Instruct (vllm, sglang) (#224)
0d61b66|2026-09-21|Rob|fixed hadolint -> broken make format (#226)
74e4375|2026-09-21|Rob|ci: skip integration test tiers for catalog-only PRs (#225)
752e268|2026-09-18|Rob|Fix FirecREST userinfo KeyError and add --qos support (#221)
df34f4e|2026-09-05|Aryan Ahadinia|Upgrade OpenTela (#216)
e957532|2026-09-05|Rob|Serialize keyring writes across xdist workers (#214)
909026a|2026-08-27|Rob|Idempotent job submission: named launches, adopt-before-retry, retry 408/429 (#211)
c66361a|2026-08-25|Aryan Ahadinia|Add Check for Model Existence (#210)
5844ccd|2026-08-19|Imanol Schlag|Add Qwen/Qwen3.5-27B (vllm) (#208)
fa000af|2026-08-13|Aryan Ahadinia|Fix Docker Image Build in CI (#207)
18b0744|2026-08-13|Aryan Ahadinia|Switch to Service Account for CI (#206)
03b44ac|2026-08-11|Aryan Ahadinia|Upgrade Dependencies and Minor Edits in Docs (#203)
91eb4fc|2026-08-11|Aryan Ahadinia|Add "Adding New Model" Instructions (#202)
614321c|2026-08-11|Aryan Ahadinia|Add Framework to SML Instructions (#201)
a81e2c3|2026-08-11|Aryan Ahadinia|Add Instructions for Building Docker Images and Using CI Pipeline in Handbook (#200)
ca80ad5|2026-08-07|Aryan Ahadinia|Rename API Key (#197)
275a05b|2026-08-06|Aryan Ahadinia|Add Namespacing to SML (#196)
8ac7cea|2026-07-24|Aryan Ahadinia|Add swiss-ai/Apertus-v1.5 Commands (#189)
c747be5|2026-07-23|Aryan Ahadinia|Fix Commands to Run (#188)
b00ef56|2026-07-23|Aryan Ahadinia|Drop Tool Parser in Thinking Apertus 1.5 Thinking (#187)
0d4d660|2026-07-22|Aryan Ahadinia|Drop Uncessecary Flags (#186)
3eea618|2026-07-22|Aryan Ahadinia|Fix the Chat Template Format (#185)
1734862|2026-07-22|Anunay Yadav|Apertus 1.5 Release Docker Image (#184)
d9fa9bc|2026-07-17|yfchoco208|Add vLLM-Omni compatible vLLM image support (#183)
4a01e4f|2026-07-16|Rob|Robmsmt/vllm v0.19 bos token (#173)
eee4d07|2026-07-16|Aryan Ahadinia|Fix CLI argument from --slurm-nodes to --nodes in multiple example scripts (#182)
b3d5d0d|2026-07-14|Aryan Ahadinia|Update RC Models (#180)
3b1a0b7|2026-07-14|Aryan Ahadinia|Update Apertus v1.5 RC Launch Scripts (#179)
28b12f3|2026-07-13|Aryan Ahadinia|Add Apertus v1.5 RC Launch Scripts (#178)
7e6e340|2026-07-13|Aryan Ahadinia|Keep the resolved env file path absolute and out of the package dir (#176)
334ccbf|2026-07-13|Aryan Ahadinia|Add Apertus Pre-release Launch Commands (#177)
bcec9c2|2026-07-10|Aryan Ahadinia|Publish PR image builds to a pr-<N> channel, not :latest (#174)
627e89a|2026-07-10|Aryan Ahadinia|Stop the live display querying widgets after teardown (#175)
cb89503|2026-07-05|Aryan Ahadinia|Add Apertus 1.5 8B Launch Commands
f36541e|2026-06-29|Aryan Ahadinia|Update Dependencies
bab07a2|2026-06-29|Aryan Ahadinia|Add Long Context Apertus 70b Test Checkpoint (#166)
d453441|2026-06-26|Aryan Ahadinia|Add Apertus 1.5 Low Less Refuse Feedback
0f917e7|2026-06-26|Aryan Ahadinia|Fix Logo Size in README
97918c0|2026-06-26|Aryan Ahadinia|Add New Logo
459909e|2026-06-23|Aryan Ahadinia|Add Apertus 1.5 (Aligned for Less Refusal Rate) Launch Script (#163)
4413441|2026-06-23|Aryan Ahadinia|Fix SGL Router Registration Issue | Opening Node Terminal in TUI (#161)
5ae438e|2026-06-19|Copilot|Fix SonarCloud scan failing on fork PRs due to missing SONAR_TOKEN (#151)
95a0d80|2026-06-19|Aryan Ahadinia|Add Consecutive Job Submission (#159)
6bb04cb|2026-06-19|robmsmt|add resp parser; and chat template for think+tool
d84ffe3|2026-06-19|Aryan Ahadinia|Add DeepSeek-V4-Flash (#160)
2d2f093|2026-06-19|robmsmt|comment
e30cde6|2026-06-19|robmsmt|add apertus reasoning parser
1cfdbf4|2026-06-19|robmsmt|RL think+TOOL; plus drop old tokenizer path
722ebdf|2026-06-19|Aryan Ahadinia|Add Replica-specific Health Check (#142)
c64a0f8|2026-06-18|Rob|Build images for amd64 + arm64 and merge into multi-arch manifest (#157)
c4a1bcb|2026-06-18|Anunay Yadav|Apertus vllm serving image: Commit hashes pin  + hf cache directory c… (#156)
935f707|2026-06-18|Matéo - subnet.dev|Enable multi platform on vllm_apertus_1.5 Dockerfile (#152)
1dcf0ee|2026-06-18|robmsmt|fix
d3b27d7|2026-06-18|robmsmt|added think as seperate model
3e32a10|2026-06-18|robmsmt|apertus 1.5 - add mixthink vs nothink 1606_480it
3adc49d|2026-06-17|robmsmt|add 1.5 rl mix think
635558a|2026-06-17|robmsmt|add rl step260 example
05ee56f|2026-06-08|Burak Emir Sezen|K6 Integration for Loadtesting (#110)
0036a27|2026-06-08|robmsmt|add rl mock
0ed2026|2026-06-03|robmsmt|add apertus 1.5 sft (no dpo) as example
8a8471f|2026-06-03|Rob|added apertus1.5 8b sft dpo example (#154)
c1bce7c|2026-06-03|dtamayo|doc: Add long context checkpoint (#153)
6feb968|2026-05-29|Burak Emir Sezen|Configure max tokens for model health check (#150)
e5dea70|2026-05-28|Rob|Include model_launch_version in logs payload (#149)
5931ff3|2026-05-28|Burak Emir Sezen|Option to change slurm account (#148)
```

## Last 100 commits: co-author names per commit (from bodies)
```
02b74ce Rob ['model-launch-factory'] 
6f8bea0 Rob ['Claude Fable 5.1', 'model-launch-factory'] Claude-Session
898b358 Rob ['model-launch-factory'] 
001a3ac Faruk Zahiragić ['Claude Opus 5.5'] 
dc10d59 Rob ['Claude Opus 5.5'] 
521da02 Aryan Ahadinia ['Claude Opus 5.5 (1M context)'] 
c6c0e87 Aryan Ahadinia ['Claude Opus 5.5', 'Claude Opus 5.5 (1M context)'] 
405e54d Aryan Ahadinia ['Claude Opus 5.5 (1M context)'] 
5d1f50b Mel Zuercher ['Claude Opus 5.5'] 
1b3f599 Rob ['Claude Fable 5.1', 'model-launch-factory'] Claude-Session
6f1ba2a Rob ['model-launch-factory'] 
bcfe0e8 Rob ['model-launch-factory'] 
1c12244 Rob ['model-launch-factory'] 
1c0c7c4 Rob ['model-launch-factory'] 
d9dd73e Rob ['Claude Fable 5.1'] Claude-Session
9ce8ffb Rob ['Claude Fable 5.1'] Claude-Session
970a965 Rob ['Claude Fable 5.1', 'model-launch-factory'] Claude-Session
1256c9d Rob ['Claude Fable 5.1', 'model-launch-factory'] Claude-Session
180c8d2 Rob ['Claude Fable 5.1'] Claude-Session
586aa7d Rob ['Claude Fable 5.1'] Claude-Session
871f862 Rob ['Claude Fable 5.1'] Claude-Session
719e9f3 Rob ['Claude Fable 5.1'] Claude-Session
b14abe0 Rob ['Claude Fable 5.1'] Claude-Session
403a7b9 Rob ['Claude Fable 5.1'] Claude-Session
ff8d4e9 Rob ['Claude Fable 5.1'] Claude-Session
6836925 Rob ['Claude Fable 5.1'] Claude-Session
b7d63e5 Rob ['Claude Fable 5.1'] Claude-Session
ecfe37f Rob ['Claude Fable 5.1'] Claude-Session
364e8c1 Rob ['Claude Fable 5.1'] Claude-Session
ae7497a Rob ['model-launch-factory'] 
d901a07 Rob ['Claude Fable 5.1'] Claude-Session
05c4854 Rob ['Claude Fable 5.1'] Claude-Session
3df4362 Rob ['model-launch-factory'] 
74e4375 Rob ['Claude Sonnet 5'] Claude-Session
752e268 Rob ['Claude Sonnet 5'] Claude-Session
e957532 Rob ['Claude Fable 5'] 
909026a Rob ['Claude Fable 5'] 
1734862 Anunay Yadav ['Aryan Ahadinia', 'Copilot Autofix powered by AI'] 
4a01e4f Rob ['Aryan Ahadinia'] 
7e6e340 Aryan Ahadinia ['Claude Opus 4.8'] Claude-Session
bcec9c2 Aryan Ahadinia ['Claude Opus 4.8'] Claude-Session
627e89a Aryan Ahadinia ['Claude Opus 4.8'] Claude-Session
bab07a2 Aryan Ahadinia ['Daniel Tamayo Mela'] 
5ae438e Copilot ['copilot-swe-agent[bot]'] 
c64a0f8 Rob ['Claude Opus 4.8 (1M context)'] 
05ee56f Burak Emir Sezen ['Aryan Ahadinia', 'Claude Opus 4.8 (1M context)', 'robmsmt'] 
```

## Last 50 non-merge commits: shortstat
```

02b74ce|Rob|examples: add deepseek-ai/DeepSeek-V4.1-Flash-1017 (vllm) (#268)   5 files changed, 186 insertions(+) 
6f8bea0|Rob|examples: add XiaomiMiMo/MiMo-V2.6-Flash-RL (+ image vllm_nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423) (#231)   7 files changed, 293 insertions(+) 
898b358|Rob|examples: add HuggingFaceTB/SmolLM2-135M-Instruct (vllm) (#260)   5 files changed, 137 insertions(+) 
72b8c1c|Rob|ci: make lightweight integration tests opt-in (#282)   3 files changed, 7 insertions(+), 4 deletions(-) 
001a3ac|Faruk Zahiragić|Add AGENTS.md requiring a dedicated branch per change (#278)   2 files changed, 19 insertions(+) 
9965ba6|Rob|Run sml on KISTI: stock pyxis container runtime, extra sbatch args, working dir (#280)   18 files changed, 597 insertions(+), 821 deletions(-) 
dc10d59|Rob|fix(launchers): give the dev OpenTela bootstrap its own peer ID (#274)   1 file changed, 6 insertions(+), 4 deletions(-) 
521da02|Aryan Ahadinia|envs: migrate tested SGLang examples to sglang_base, drop Kimi-K2 and sglang_0.5.20 (#271)   37 files changed, 51 insertions(+), 141 deletions(-) 
74da8ab|Aryan Ahadinia|Migrate vLLM examples to vllm_base (#265)   37 files changed, 31 insertions(+), 342 deletions(-) 
c6c0e87|Aryan Ahadinia|Stop building container images in model-launch (#259)   12 files changed, 35 insertions(+), 2215 deletions(-) 
405e54d|Aryan Ahadinia|cli: pass a flag-only --served-model-name to the framework (#269)   2 files changed, 21 insertions(+) 
5d1f50b|Mel Zuercher|launchers: configurable working directory (default stays ~/.sml) (#261)   5 files changed, 106 insertions(+), 9 deletions(-) 
1b3f599|Rob|examples: add Qwen/Qwen3.8-27B (#246)   3 files changed, 29 insertions(+), 19 deletions(-) 
6f1ba2a|Rob|examples: add zai-org/GLM-5.3 (vllm) (#240)   6 files changed, 244 insertions(+) 
bcfe0e8|Rob|examples: add XiaomiMiMo/MiMo-V2.6-Pro-RL (vllm) (#239)   6 files changed, 295 insertions(+) 
1c12244|Rob|examples: add zai-org/GLM-5.3-Flash (vllm) (#230)   5 files changed, 259 insertions(+) 
1c0c7c4|Rob|examples: add deepseek-ai/DeepSeek-V4.1-Flash (vllm) (#229)   5 files changed, 188 insertions(+) 
d9dd73e|Rob|ci: a release build must not die on chmod of the shared release directory (#258)   1 file changed, 3 insertions(+), 1 deletion(-) 
9ce8ffb|Rob|images: arch-aware vllm_cuda13 template (split from #235) (#257)   1 file changed, 32 insertions(+), 13 deletions(-) 
970a965|Rob|images: add sglang_0.5.20 (sglang 0.5.20) (#245)   2 files changed, 129 insertions(+) 
1256c9d|Rob|images: add vllm_0.30.0 (vllm v0.30.0) (#234)   2 files changed, 101 insertions(+) 
180c8d2|Rob|ci: manifest gate reads scan checks with GET; a tick does not rescan what already passed (#256)   1 file changed, 18 insertions(+), 2 deletions(-) 
586aa7d|Rob|ci: allow PyCryptodome's self-test keys; one image's scan no longer blocks another's manifest (#255)   2 files changed, 33 insertions(+), 4 deletions(-) 
871f862|Rob|ci: the finisher completes a check from the registry when the image is already there (#254)   1 file changed, 80 insertions(+), 3 deletions(-) 
719e9f3|Rob|ci: a cluster job pushes each finished build to GHCR with a token the finisher lends it (#253)   2 files changed, 287 insertions(+), 166 deletions(-) 
b14abe0|Rob|ci: the finisher rides out one cluster's FirecREST being down or unable to stage a download (#252)   1 file changed, 73 insertions(+), 19 deletions(-) 
403a7b9|Rob|ci: harden the image build pipeline (#250)   6 files changed, 194 insertions(+), 42 deletions(-) 
ff8d4e9|Rob|ci: the runner pushes the build's OCI archive to GHCR; the App only dispatches (#249)   2 files changed, 85 insertions(+), 4 deletions(-) 
6836925|Rob|ci: build auth as a GitHub App; remove stale context files before upload; OCI source label (#248)   3 files changed, 101 insertions(+), 21 deletions(-) 
b7d63e5|Rob|ci: image builds submit and exit; a scheduled workflow finishes, scans and merges them (#243)   3 files changed, 539 insertions(+), 217 deletions(-) 
ecfe37f|Rob|ci: build_image.py adopts its SLURM job by name and rides out FirecREST errors (#237)   1 file changed, 65 insertions(+), 10 deletions(-) 
364e8c1|Rob|examples: drop Apertus-1.5-8B sft-capfilter innovator-fix-it23409 — its checkpoint is gone from capstor (#238)   1 file changed, 16 deletions(-) 
ae7497a|Rob|catalog: add Qwen/Qwen3.8-27B (vllm, sglang) (#228)   11 files changed, 550 insertions(+) 
d901a07|Rob|ci: catalog-only PRs may add env TOMLs and image Dockerfiles (#233)   1 file changed, 11 insertions(+), 8 deletions(-) 
05c4854|Rob|ci: build_image.py reads the primary group like the launcher (FirecREST 2.6.0+) (#232)   1 file changed, 4 insertions(+), 1 deletion(-) 
3df4362|Rob|catalog: add Qwen/Qwen2.5-0.5B-Instruct (vllm, sglang) (#224)   11 files changed, 282 insertions(+) 
0d61b66|Rob|fixed hadolint -> broken make format (#226)   4 files changed, 25 insertions(+), 25 deletions(-) 
74e4375|Rob|ci: skip integration test tiers for catalog-only PRs (#225)   1 file changed, 45 insertions(+) 
752e268|Rob|Fix FirecREST userinfo KeyError and add --qos support (#221)   15 files changed, 292 insertions(+), 6 deletions(-) 
df34f4e|Aryan Ahadinia|Upgrade OpenTela (#216)   2 files changed, 7 insertions(+), 7 deletions(-) 
e957532|Rob|Serialize keyring writes across xdist workers (#214)   1 file changed, 38 insertions(+), 7 deletions(-) 
909026a|Rob|Idempotent job submission: named launches, adopt-before-retry, retry 408/429 (#211)   20 files changed, 684 insertions(+), 151 deletions(-) 
c66361a|Aryan Ahadinia|Add Check for Model Existence (#210)   31 files changed, 912 insertions(+), 197 deletions(-) 
5844ccd|Imanol Schlag|Add Qwen/Qwen3.5-27B (vllm) (#208)   2 files changed, 20 insertions(+) 
fa000af|Aryan Ahadinia|Fix Docker Image Build in CI (#207)   3 files changed, 81 insertions(+), 2 deletions(-) 
18b0744|Aryan Ahadinia|Switch to Service Account for CI (#206)   18 files changed, 328 insertions(+), 113 deletions(-) 
03b44ac|Aryan Ahadinia|Upgrade Dependencies and Minor Edits in Docs (#203)   4 files changed, 1034 insertions(+), 941 deletions(-) 
91eb4fc|Aryan Ahadinia|Add "Adding New Model" Instructions (#202)   4 files changed, 87 insertions(+), 16 deletions(-) 
614321c|Aryan Ahadinia|Add Framework to SML Instructions (#201)   3 files changed, 124 insertions(+) 
a81e2c3|Aryan Ahadinia|Add Instructions for Building Docker Images and Using CI Pipeline in Handbook (#200)   5 files changed, 212 insertions(+), 16 deletions(-) ```

## Merged/closed PRs, API page 1 (50 PRs by updated desc; raw JSON saved in scratch pulls_closed_1.json)
```
268 robmsmt merged 2026-10-18T12:19 2026-11-04T19:44 [] [] factory/deepseek-ai__deepseek-v4.1-flash-1017-1d55bf2d | examples: add deepseek-ai/DeepSeek-V4.1-Flash-1017
231 robmsmt merged 2026-09-24T07:50 2026-11-04T19:04 [] ['AryanAhadinia'] factory/xiaomimimo__mimo-v2.6-flash-rl-5c05920f | examples: add XiaomiMiMo/MiMo-V2.6-Flash-RL (+ image vllm_nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423)
260 robmsmt merged 2026-09-29T15:18 2026-11-04T18:52 [] [] factory/huggingfacetb__smollm2-135m-instruct-537f0b17 | examples: add HuggingFaceTB/SmolLM2-135M-Instruct
282 robmsmt merged 2026-11-04T18:47 2026-11-04T18:49 [] ['AryanAhadinia', 'FarukZahiragic'] ci/lightweight-opt-in | ci: make lightweight integration tests opt-in
278 FarukZahiragic merged 2026-10-29T19:36 2026-11-04T18:44 [] ['robmsmt', 'AryanAhadinia'] docs/agent-branch-rule | Add AGENTS.md requiring a dedicated branch per change
280 robmsmt merged 2026-11-03T16:37 2026-11-04T17:56 [] [] feat/kisti-pyxis | Run sml on KISTI: stock pyxis container runtime, extra sbatch args, working dir
274 robmsmt merged 2026-10-29T16:24 2026-10-29T16:27 [] [] fix/dev-bootstrap-peer-id | fix(launchers): give the dev OpenTela bootstrap its own peer ID
271 AryanAhadinia merged 2026-10-19T16:04 2026-10-19T16:11 [] ['robmsmt', 'FarukZahiragic'] examples/sglang-base-image | envs: migrate tested SGLang examples to sglang_base, drop Kimi-K2 and sglang_0.5.20
265 AryanAhadinia merged 2026-10-18T08:50 2026-10-19T16:01 [] [] examples/vllm-base-image | envs: migrate vLLM examples to vllm_base, drop superseded envs, fix snowflake, drop dead Apertus 1.5 scripts
259 AryanAhadinia merged 2026-09-27T11:20 2026-10-18T16:15 [] [] AA/stop-image-builds | Stop building container images in model-launch
269 AryanAhadinia merged 2026-10-18T12:41 2026-10-18T14:08 [] [] fix/advanced-served-name-to-framework | cli: pass a flag-only --served-model-name to the framework
264 AryanAhadinia closed 2026-10-17T16:33  [] ['robmsmt', 'FarukZahiragic'] AA/move-images-out | ci: move image builds to model-launch-images
261 mzueri merged 2026-10-07T12:16 2026-10-07T12:56 [] ['robmsmt'] configurable-working-dir | launchers: configurable working directory (default stays ~/.sml)
220 FarukZahiragic closed 2026-09-15T13:53  [] [] factory/huggingfacetb__smollm2-135m-instruct-ffd73be6 | catalog: add HuggingFaceTB/SmolLM2-135M-Instruct
219 FarukZahiragic closed 2026-09-15T13:26  [] [] factory/huggingfacetb__smollm2-135m-instruct-c52187a8 | catalog: add HuggingFaceTB/SmolLM2-135M-Instruct
218 FarukZahiragic closed 2026-09-15T12:42  [] [] factory/huggingfacetb__smollm2-135m-instruct-f695f533 | catalog: add HuggingFaceTB/SmolLM2-135M-Instruct
217 FarukZahiragic draft-closed 2026-09-15T11:52  [] [] factory/huggingfacetb__smollm2-135m-instruct-6d321ba9 | [TEST/DRAFT] catalog: add HuggingFaceTB/SmolLM2-135M-Instruct (factory e2e test — do not merge)
246 robmsmt merged 2026-09-24T17:18 2026-09-28T13:53 [] [] factory/qwen__qwen3.8-27b-1ecd11f6 | examples: add Qwen/Qwen3.8-27B
240 robmsmt merged 2026-09-24T14:22 2026-09-28T13:53 [] [] factory/zai-org__glm-5.3-1a8830a7 | examples: add zai-org/GLM-5.3
239 robmsmt merged 2026-09-24T14:09 2026-09-28T13:53 [] [] factory/xiaomimimo__mimo-v2.6-pro-rl-3a1dee3d | examples: add XiaomiMiMo/MiMo-V2.6-Pro-RL
230 robmsmt merged 2026-09-24T07:38 2026-09-28T13:53 [] [] factory/zai-org__glm-5.3-flash-cfc65fc1 | examples: add zai-org/GLM-5.3-Flash
229 robmsmt merged 2026-09-23T04:19 2026-09-28T13:53 [] [] factory/deepseek-ai__deepseek-v4.1-flash-22d2152f | examples: add deepseek-ai/DeepSeek-V4.1-Flash
195 youssef62 closed 2026-08-04T16:27  [] [] servekit-intergration-demo | feat: Add servekit integration for fast weight loading on sglang with GLM5.1-FP8 and GLM4.7
258 robmsmt merged 2026-09-25T19:31 2026-09-26T08:26 [] ['AryanAhadinia', 'FarukZahiragic'] ci/release-dir-chmod | ci: a release build must not die on chmod of the shared release directory
257 robmsmt merged 2026-09-25T19:19 2026-09-25T19:27 [] ['AryanAhadinia', 'FarukZahiragic'] images/arch-aware-vllm-cuda13 | images: arch-aware vllm_cuda13 template (split from #235)
245 robmsmt merged 2026-09-24T17:18 2026-09-25T19:21 [] ['FarukZahiragic'] factory/image-sglang_0.5.20 | images: add sglang_0.5.20 (sglang 0.5.20)
234 robmsmt merged 2026-09-24T12:04 2026-09-25T19:07 [] ['FarukZahiragic'] factory/image-vllm_0.30.0 | images: add vllm_0.30.0 (vllm v0.30.0)
204 AryanAhadinia closed 2026-08-11T22:20  ['images', 'requires-std-tests'] ['robmsmt'] AA/vllm-new-image | Create New Base Docker Image for vLLM
256 robmsmt merged 2026-09-25T18:00 2026-09-25T18:25 [] ['AryanAhadinia', 'FarukZahiragic'] ci/manifest-gate-get | ci: manifest gate reads scan checks with GET; a tick does not rescan what already passed
255 robmsmt merged 2026-09-25T16:13 2026-09-25T17:46 [] ['AryanAhadinia', 'FarukZahiragic'] ci/scan-cryptodome | ci: allow PyCryptodome's self-test keys; scan verdicts per image
254 robmsmt merged 2026-09-25T11:44 2026-09-25T15:46 [] ['AryanAhadinia'] ci/finish-from-ghcr | ci: the finisher completes a check from the registry when the image is already there
253 robmsmt merged 2026-09-25T09:19 2026-09-25T09:45 [] ['AryanAhadinia', 'FarukZahiragic'] ci/push-job | ci: a cluster job pushes each finished build to GHCR with a token the finisher lends it
252 robmsmt merged 2026-09-25T06:36 2026-09-25T07:58 [] ['AryanAhadinia', 'FarukZahiragic'] ci/finisher-site-outage | ci: the finisher rides out one cluster's FirecREST being down or unable to stage a download
250 robmsmt merged 2026-09-24T22:04 2026-09-25T06:30 [] ['AryanAhadinia', 'FarukZahiragic'] ci/audit-fixes | ci: harden the image build pipeline
249 robmsmt merged 2026-09-24T21:06 2026-09-24T21:29 [] [] ci/push-from-runner | ci: the runner pushes the build's OCI archive to GHCR; the App only dispatches
248 robmsmt merged 2026-09-24T19:35 2026-09-24T20:40 [] [] ci/build-upload-truncate-and-oci-label | ci: build auth as a GitHub App; remove stale context files before upload; OCI source label
243 robmsmt merged 2026-09-24T16:44 2026-09-24T17:51 [] [] ci/async-image-builds | ci: image builds submit and exit; a scheduled workflow finishes, scans and merges them
247 robmsmt closed 2026-09-24T17:35  [] [] ci/never-auto-rebuild-release-images | ci: never rebuild *_release* images by detection — only a dispatch naming the image
244 robmsmt closed 2026-09-24T17:09  [] [] ci/scan-allowlist-package-tests | ci: image-scan allowlist — installed packages' test suites and botocore service data
237 robmsmt merged 2026-09-24T13:17 2026-09-24T16:38 [] [] ci/build-image-adopt-by-name | ci: build_image.py adopts its SLURM job by name and rides out FirecREST errors
242 robmsmt draft-closed 2026-09-24T16:24  [] [] factory/qwen__qwen3.8-27b-a18b4e4c | examples: add Qwen/Qwen3.8-27B
241 robmsmt draft-closed 2026-09-24T16:11  [] [] factory/qwen__qwen3.8-27b-a18b4e4c | examples: add Qwen/Qwen3.8-27B
238 robmsmt merged 2026-09-24T13:28 2026-09-24T13:34 [] [] examples/drop-innovator-fix-it23409 | examples: drop Apertus-1.5-8B innovator-fix-it23409 (checkpoint gone from capstor)
236 robmsmt draft-closed 2026-09-24T13:16  [] [] factory/qwen__qwen3.8-27b-a18b4e4c | examples: add Qwen/Qwen3.8-27B
228 robmsmt merged 2026-09-22T23:01 2026-09-24T11:58 [] [] factory/qwen__qwen3.8-27b-a18b4e4c | catalog: add Qwen/Qwen3.8-27B
233 robmsmt merged 2026-09-24T11:35 2026-09-24T11:44 [] [] ci/catalog-only-envs-images | ci: catalog-only PRs may add env TOMLs and image Dockerfiles
232 robmsmt merged 2026-09-24T07:53 2026-09-24T10:01 [] [] fix/build-image-userinfo-group | ci: build_image.py reads the primary group like the launcher (FirecREST 2.6.0+)
224 robmsmt merged 2026-09-21T17:54 2026-09-21T18:29 [] [] factory/qwen__qwen2.5-0.5b-instruct-72dfde7e | catalog: add Qwen/Qwen2.5-0.5B-Instruct
226 robmsmt merged 2026-09-21T18:25 2026-09-21T18:27 [] ['AryanAhadinia'] fix/hadolint-py-wheel-corruption | fix: pin hadolint-py to unbreak make format
225 robmsmt merged 2026-09-21T18:05 2026-09-21T18:15 [] [] fix/skip-integration-tests-for-catalog-only-prs | ci: skip integration test tiers for catalog-only PRs
```

## Full-history trailers
Signed-off-by: 2 commits, both dependabot[bot] (d99428c, cb0132a). Claude co-author: 35 commits (Rob 26, Aryan 6, Burak 1, Faruk 1, Mel 1). Claude-Session: 24 commits. First Claude co-author trailer: 7dfaac9 (28 May 2026) by log order, 05ee56f is 8 Jun.

## Merge commits (8, all Dec 2025 to Mar 2026)
```
b70a1f7 2026-03-09 Davit Melikidze Merge pull request #13 from swiss-ai/nccl_fix
b363067 2026-03-02 Rob Merge pull request #8 from xzyaoi/patch-1
0b02dab 2025-12-22 Rob Merge pull request #5 from swiss-ai/dedeswim-patch-1
6065b26 2025-12-17 Rob Merge pull request #4 from xzyaoi/patch-1
7881553 2025-12-15 Rob Merge pull request #3 from swiss-ai/opencode-instructions
67ed3b0 2025-12-10 Rob Merge pull request #2 from swiss-ai/robmsmt/various
f911d74 2025-12-09 robmsmt Merge branch 'main' of https://github.com/swiss-ai/model-launch
1f43af0 2025-12-09 Rob Merge pull request #1 from lsnnova/patch-1
```
