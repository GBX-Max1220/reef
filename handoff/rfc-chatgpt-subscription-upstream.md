# [RFC] A ChatGPT subscription as the upstream of a local Reef

<!-- Draft for review; fields follow .github/ISSUE_TEMPLATE/rfc.yml. -->

### Primary area

Multiple areas (Service, API, or CLI; Harness, recipes, or surfaces)

### Summary and decision to make

Many people who want to try Reef, Reefine in particular, have a ChatGPT Plus or Pro subscription and no API key. Reef's upstream today is a URL and an API key. This RFC proposes an opt-in upstream in which a **local, single-user** Reef signs in with the person's own ChatGPT account through OpenAI's sign-in, keeps and refreshes the token itself, and serves every model call through Reef as it does today, so records, Reefine and evaluation keep working.

Decisions requested:

1. Whether Reef may hold a consumer subscription login at all, limited to a deployment that listens on loopback for one person.
2. The one exception to "preserve provider request bodies" this upstream needs: the ChatGPT backend accepts only a constrained Responses body (streamed, not stored, system text in `instructions`).
3. Whether a single-component recipe on this upstream sends its evaluation episodes and proposer calls back through the service's evaluation route, as a composite recipe already does (`reef/service/assembly.py:294-304`), so the token never leaves the Reef process.

Claude subscriptions are out of scope. Anthropic's terms say that it does "not permit third-party developers to ... route requests through Free, Pro, or Max plan credentials on behalf of their users" and that developers "may not collect, store, or intermediate Claude.ai credentials or session tokens" ([Claude Code legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)). A Claude subscription can only be used by the unmodified Claude Code binary talking to Anthropic itself; how Reef could still version and evolve that harness with Reef outside the model path is a separate discussion.

### Motivation and supporting results

- The upstream is `inference.upstream_url` plus `inference.upstream_api_key`, in one of three dialects (`PROVIDER_APIS = ("openai", "responses", "anthropic")`, `reef/inference/http.py:41`). `ProviderRequestHeaders` adds a static key (`reef/inference/http.py:72-91`). Nothing can sign in or refresh.
- pi can already sign in to ChatGPT (`@earendil-works/pi-ai`, provider `openai-codex`, base `https://chatgpt.com/backend-api`), but `reef-pi` sends every call through Reef's `reef` provider, and Reef can only present an API key. A session that used pi's own login instead would leave Reef: no records, Reefine evaluating a different model, and against the rule that every model call stays on Reef's binding (#618, #632).
- The silent version of that bypass existed until the pi `defaultModel` fix (PR pending): pi 0.84.2 never resolved `reef/{model}` and fell back to any provider it held a credential for, including a `/login`.
- OpenAI has not restricted ChatGPT sign-in in third-party agents, and several harnesses offer it; I found no written permission, only the absence of a restriction (see Risks).
- In a Reefine run on a single-component recipe, evaluation episodes call the upstream directly with its key and a minimal environment (`ModelBinding.from_runtime`, `reef/harness/episodes/model_binding.py:114-127`; `reef/harness/episodes/executor.py:152-158`). Behind a TLS-intercepting proxy the episode's pi fails with `SELF_SIGNED_CERT_IN_CHAIN`, reported as "Connection error.", and the step is rejected as though the request were at fault. Routing episodes through the service (decision 3) removes that failure along with the token exposure.

### Goals and non-goals

Goals:

- `reef-pi` sessions, the Reefine text and agent proposers, and evaluation episodes all run on the person's ChatGPT plan through Reef.
- Sign-in goes through OpenAI's own flow; Reef never asks for a password.
- The token lives only in the Reef process and one file readable by its owner; no tree, episode, record, page or log carries it.
- Plan limits surface as plan limits, not as failed tasks.

Non-goals:

- Claude Free, Pro or Max plans (terms forbid it; see above).
- A hosted or shared Reef serving other people on one person's plan: that is reselling usage.
- Chat Completions or Anthropic-dialect clients on this upstream; it serves the Responses dialect only.
- Other subscription sign-ins pi supports (GitHub Copilot, Kimi, xAI); each would follow its provider's terms in its own proposal.
- Weight recipes: the backend returns text, not the token ids and log probabilities training needs.

### Proposal

1. **Sign-in.** `reef login chatgpt` runs OpenAI's device-code sign-in, the flow the Codex CLI offers with `codex login --device-auth`: it prints a URL and a code, the person approves in a browser on any device, and Reef stores the access token, refresh token, expiry and account id in `~/.reef/credentials/chatgpt.json` with mode 0600. `reef logout chatgpt` deletes it.
2. **Upstream.** `inference.upstream_api: chatgpt` needs no URL or key; the base is `https://chatgpt.com/backend-api`. A `SubscriptionRequestHeaders(RequestHeadersFactory)` sends `Authorization: Bearer <access token>`, `chatgpt-account-id`, `originator` and `OpenAI-Beta: responses=experimental`, refreshes before expiry, and on a 401 refreshes once and retries.
3. **Path and body.** Clients call `/v1/responses` as they do on the `responses` dialect; Reef maps it to `/codex/responses` and adjusts the body at this upstream only: `stream: true`, `store: false`, system and developer input items moved into `instructions`, `include: ["reasoning.encrypted_content"]`, and parameters the backend refuses removed. The record keeps the client's request as sent; the adjustments are listed in the record's metadata. A non-streaming request, from a client or from Reef itself, is streamed upstream and folded into one response. The fold takes the output from the `response.output_item.done` events: with `store: false` the backend's `response.completed` carries an empty `output` (observed in the spike), and `_fold_responses_stream` (`reef/harness/episodes/model_binding.py:563-591`) returns that event as it is, so without this change Reef's own plan, answer and review calls would read empty replies.
4. **Clients.** The upstream presents the `responses` dialect, so the pi binding renders `openai-responses` and nothing changes on the client side. Codex speaks Responses as well; adapters bound through the Anthropic dialect are refused at startup with the reason.
5. **Evaluation and proposer calls.** On this upstream a single-component recipe gets the served endpoint a composite gets today, so `CordisRecipe.model_binding` returns the evaluation route (`reef/recipe/cordis.py:646-678`) and episodes, the text proposer and the agent proposer's gateway reach the plan through Reef with the evaluation token.
6. **Models.** `inference.upstream_model` must be a model the signed-in plan offers; startup checks it with one small call and names the refusal when it fails. A static list is not enough: in the spike a ChatGPT Plus/Pro account accepted `gpt-5.5` and `gpt-5.6-luna` and refused `gpt-5.4`, `gpt-5.4-mini` and `gpt-5.3-codex-spark` ("not supported when using Codex with a ChatGPT account"), although pi-ai's catalog for this backend lists all five.
7. **Scope check.** Startup refuses `upstream_api: chatgpt` when `reef.host` is not a loopback address.

### Public interfaces and configuration

- CLI: `reef login chatgpt`, `reef logout chatgpt`; `reef serve --recipe reefine --inference.upstream-api chatgpt --inference.upstream-model <model>`.
- Configuration: `inference.upstream_api` gains `chatgpt`; `inference.upstream_url` and `inference.upstream_api_key` are refused with it.
- Files: `~/.reef/credentials/chatgpt.json` (0600).
- Wire contracts to clients: unchanged.

### Compatibility and migration

Additive. The three existing dialects, the pi binding, records and the evaluation route are unchanged; a deployment that does not select `chatgpt` behaves as today.

### Security, privacy, and trust boundaries

- Reef holds a consumer credential: one owner-readable file and the service process. It is never rendered into a tree or an episode, never written to records, step records, pages or logs, and the startup settings printout masks it as it masks `upstream-api-key`.
- Evaluation episodes and the agent proposer receive the evaluation token, which the app accepts on the evaluation routes alone.
- Loopback only; a deployment serving other people cannot select this upstream.
- Sign-in completes in OpenAI's own page; Reef sees only the resulting tokens.

### Operations and observability

- Startup confirms the sign-in and the model, and names `reef login chatgpt` when either fails.
- A plan usage limit reaches the client with the backend's own message and status. A Reefine step whose episodes hit the limit reports that the evaluation could not run, not a score of 0.
- `reef-pi doctor` reports the sign-in state.

### Testing and documentation

- Hermetic tests against a stub backend: headers, path mapping, each body adjustment, stream folding, refresh before expiry and after a 401, the loopback check, and a token that never appears in rendered trees, records or logs.
- An opt-in smoke test with a signed-in account, skipped by default like the `REEF_REAL_*` smokes.
- Documentation: configuration reference, the Reefine guide, and one README line for people with a plan and no key.

### Alternatives considered

- **pi's own ChatGPT sign-in, bypassing Reef.** Works today, but Reef records nothing, Reefine evaluates another model, and it breaks the binding rule of #618 and #632.
- **Reuse the Codex CLI's sign-in file.** One step fewer for Codex users, but it ties Reef to another tool's file format, and both would refresh the same token. Possible later as an import option.
- **A third-party process that wraps a ChatGPT sign-in as an OpenAI-compatible endpoint.** Code outside the project would hold the credential, with nothing for Reef to test or own.

### Risks and unresolved questions

- **Terms.** OpenAI tolerates ChatGPT sign-in in third-party agents today, without a written permission I could find. It could change; the feature must fail clearly when the backend refuses. Should a maintainer confirm with OpenAI before this ships?
- **The backend is not a public API.** Its body rules can change without notice; the smoke test pins what Reef relies on.
- **Plan limits.** Evaluation episodes and the agent proposer spend the person's plan. Should the Reefine profile default to the text proposer or a smaller episode budget on this upstream?
- **Records.** Records stay local and feed harness evolution only; weight recipes are out of scope. Is that the right line?
- **Client identity.** Which OAuth client id and `originator` Reef presents at sign-in.

### Implementation plan and ownership

0. Spike, before this RFC leaves draft. Done so far, with a device-code sign-in through the pinned Codex CLI and a loopback script applying the proposed headers and body adjustments: the backend answered streamed and folded requests with a pi-shaped Responses body, and the two findings above (empty `output` on `response.completed`, plan-dependent models) came from it. Still to run: `reef-pi` and one Reefine request through Reef against it.
1. Sign-in, credential file, `SubscriptionRequestHeaders`, path and body adjustment on the streaming path: `reef-pi` sessions work.
2. Single-component recipes route evaluation and proposer calls through the evaluation route on this upstream; non-streaming folding: Reefine works end to end.
3. Startup and doctor checks, plan-limit reporting, documentation.

Owner: @simonucl. Reviews: @Benjamin-eecs for the model binding and #632. Related: #204, #618, #632.

### Platforms

macOS, Linux and WSL 2 alike; the device-code sign-in needs no local browser.

### Submission checks

- [x] I searched existing issues, pull requests, and RFCs for this decision.
- [x] I understand this issue is the RFC and that discussion is not acceptance.
- [x] I will keep this issue updated with material design changes and implementation links.
