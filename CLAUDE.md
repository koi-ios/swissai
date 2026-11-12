@AGENTS.md

## Claude Code only

- Use subagents to read and search; make edits in the main session.
- Auto memory is per machine and per tool. Anything durable goes into a file in this repository.
- `.claude/settings.json` keeps instruction files under `vendor/` from loading, and denies `git push`, `gh` and Modal launches. For work inside a clone, or to run upstream code, start the session with `claude --settings pipeline/sandbox.settings.json`.
