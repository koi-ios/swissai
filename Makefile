# swissai control plane. A menu: every target is one line that calls a file in scripts/.
# Works with the make that ships with macOS (3.81). Method: pipeline/DOCTRINE.md.

PY ?= python3
.DEFAULT_GOAL := help
.PHONY: help status portfolio check test clones clones-verify budget spend export leakcheck new-project backup research

help:  ## list the targets
	@grep -E '^[a-z][a-z-]*:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

status:  ## the portfolio on one screen
	@$(PY) scripts/portfolio.py

portfolio:  ## rewrite PORTFOLIO.md from the project cards and the budget
	@$(PY) scripts/portfolio.py --write

check:  ## lint the control plane (STRICT=1 makes warnings fail)
	@$(PY) scripts/check.py $(if $(STRICT),--strict,)

test:  ## rerun test suites against the recorded counts (PROJECT=06 for one project)
	@$(PY) scripts/test.py $(PROJECT)

clones:  ## every clone and worktree, its role and state
	@$(PY) scripts/clones.py

clones-verify:  ## does each project's clone equal its tracked patches?
	@$(PY) scripts/clones.py verify

budget:  ## may a run be launched? PROJECT=02 COST=6
	@$(PY) scripts/budget.py gate --project $(PROJECT) --cost $(COST)

spend:  ## record a run: PROJECT=02 STEP="gpu_check" ACTUAL=5.40 [GPU= SECONDS= PLANNED= APP= NOTE=]
	@$(PY) scripts/budget.py spend --project $(PROJECT) --step "$(STEP)" --actual $(ACTUAL) --gpu "$(GPU)" --seconds $(or $(SECONDS),0) --planned $(or $(PLANNED),0) --app-id "$(APP)" --note "$(NOTE)"

export:  ## (user) prepare a series for upstream: PROJECT=06 [REPO=OpenTela]. Never pushes
	@$(PY) scripts/export.py $(PROJECT) $(if $(REPO),--repo $(REPO),) $(ARGS)

leakcheck:  ## (user) leak-check an exported worktree: DIR=... REPO=OpenTela BASE=origin/main
	@$(PY) scripts/leakcheck.py $(DIR) --repo $(REPO) --base $(or $(BASE),origin/main)

new-project:  ## a new parked project from the template: SLUG=10-short-name
	@bash scripts/new_project.sh $(SLUG)

backup:  ## (user) write a git bundle of this repository: DEST=/an/existing/folder
	@bash scripts/backup.sh $(DEST)

research:  ## rebuild the generated tables of pipeline/RESEARCH.md
	@$(PY) pipeline/build_research.py
