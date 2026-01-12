# RalphBoost (Code-First Reset)

RalphBoost is being rebuilt as a code-only, executable-verification loop.
The goal is a boosting-style TDD engine: each iteration fixes the remaining
test failures with minimal patches, and completion only happens when tests pass.

Project Map
- [HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md)
- [ONBOARDING_CHAT.md](ONBOARDING_CHAT.md)

Start here
- `docs/CODE_FIRST_HANDOFF.md`
- `docs/CODE_FIRST_ARCHITECTURE.md`
- `docs/CODE_FIRST_TDD.md`
- `docs/CHALLENGE_SUITE.md`
- `docs/TEAM_STARTER.md`

Benchmarks
- `benchmarks/README.md`
- Tiered task specs under `benchmarks/tier1`, `benchmarks/tier2`, `benchmarks/tier3`

Legacy (v1 doc-loop artifacts)
- Old docs and outputs are in `docs/legacy/`
- Old tasks are in `legacy/tasks_v1/`

Repo goals (near term)
- Code-only contract and patch engine
- Executable verifier (tests = truth)
- Telemetry of loss, thrash, patch size, and convergence

