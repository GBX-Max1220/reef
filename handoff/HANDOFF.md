# Handoff: Reef harness work (continue in a local session)

Files in this directory: `HANDOFF.md` (this note), `chatgpt_bridge.py` (the spike bridge) and
`rfc-chatgpt-subscription-upstream.md` (the RFC draft). This directory is a handoff, not project code:
drop its commit before opening the pi fix PR.

Context for a local Claude Code session in my clone of `Human-Agent-Society/reef`
(GitHub: simonucl). Picks up from a cloud session on 2026-09-27.

## Done

- **pi `defaultModel` fix**, pushed to branch `claude/gifted-einstein-wkzgx0` (commit `c8f8896`), no PR yet.
  - Bug: `reef/harness/adapters/pi/descriptor.yaml` wrote `defaultModel: "reef/{model}"`; pi 0.84.2 looks it
    up as `getModel(defaultProvider, defaultModel)` (exact id), so it always missed and pi fell back to the
    first provider it held a credential for (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, AWS keys, a `/login`).
    A `reef-pi` session then silently skipped Reef. Evaluation episodes were unaffected (minimal env).
  - Fix: bare `{model}`. Tests updated; new real-pi smoke test
    `tests/smoke/test_real_pi.py::test_real_pi_keeps_the_bound_model_beside_other_credentials`
    (fails before, passes after; needs `REEF_REAL_PI_BINARY`).
  - Next: open the PR with `.github/PULL_REQUEST_TEMPLATE.md`, ask @Benjamin-eecs to review (overlaps #632).

## Other findings (not yet filed)

1. `reef/inference/http.py:170-185`: on the streaming path an upstream error body is read raw with
   `auto_decompress=False`, so a gzip error reaches the agent as bytes instead of the provider's message.
2. Evaluation episodes of a single-component recipe (Reefine) call the upstream directly with a minimal env
   (`reef/harness/episodes/executor.py:152`, `reef/service/assembly.py:294-304`). Behind a TLS-intercepting
   proxy pi fails with `SELF_SIGNED_CERT_IN_CHAIN` -> "Connection error.", and the step is rejected with
   "rephrase or split the request". Related to my #204.
3. Reefine's review (`reef/recipe/reefine/agent.py:686`, `evolution.py:1054`) sees only the design text,
   not the agent's recorded trial results (`agent-checks.jsonl`); a proven change was rejected because
   `design.md` still said "I will check". Small first step toward Bo's #623.

## ChatGPT subscription as Reef's upstream (RFC draft: `rfc-chatgpt-subscription-upstream.md`)

Claude subscriptions are out (Anthropic terms forbid intermediating them). ChatGPT via OpenAI's own sign-in is
the target. Spike so far, with `codex login --device-auth` and `chatgpt_bridge.py` (loopback bridge that adds
auth headers and adjusts the body):

- The backend answered streamed (pi-shaped) and folded requests.
- A ChatGPT account accepted `gpt-5.5`, `gpt-5.6-luna`; refused `gpt-5.4`, `gpt-5.4-mini`,
  `gpt-5.3-codex-spark`, though pi-ai's catalog lists them.
- With `store: false`, `response.completed` has an empty `output`; items come in
  `response.output_item.done`. Reef's `_fold_responses_stream`
  (`reef/harness/episodes/model_binding.py:563-591`) returns the completed event as-is, so Reef's own
  plan/answer/review calls would read empty replies.

**Still to run (locally):** Reef in front of the bridge.

```bash
# setup, once (from AGENTS.md)
git fetch origin && git checkout claude/gifted-einstein-wkzgx0
git submodule update --init --recursive
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]" -e ./third_party/reef-client

# ChatGPT sign-in through OpenAI's own client, then the bridge (needs aiohttp)
codex login
CODEX_HOME=~/.codex python handoff/chatgpt_bridge.py &

# Reef on the bridge; state goes under ./.reef/reefine
REEF_UPSTREAM_URL=http://127.0.0.1:9902 REEF_UPSTREAM_MODEL=gpt-5.5 \
REEF_UPSTREAM_API_KEY=unused REEF_PROPOSER_SANDBOX=none \
  reef serve --recipe reefine --inference.upstream-api responses

# another terminal, same venv
curl -fsS -H "Content-Type: application/json" -d '{"name": "chatgpt-harness"}' http://127.0.0.1:8901/reef/scenarios
curl -fsS -H "x-reef-scenario: chatgpt-harness" 'http://127.0.0.1:8901/reef/harness/install?adapter=pi' | bash
./reef-harness/reef-pi -p "Run the shell command 'echo reef-ok' and reply with its output only."
./reef-harness/reef-pi evolve "when I ask you to fix a bug, reproduce it with a failing test first" --wait
```

Expect the Reefine plan/answer/review calls to come back empty until `_fold_responses_stream` takes output
from the item events. That is the first code change of the RFC's phase 1. `REEF_PROPOSER_SANDBOX=none`
runs the agent proposer unisolated; fine on your own machine, but on Linux with `bwrap` and `pasta` you can
leave it unset.

Record the outcome in the RFC's "Implementation plan" step 0, then post the RFC with
`.github/ISSUE_TEMPLATE/rfc.yml` and ping @Benjamin-eecs.
