# spade

**Status: beta.** SPADE remains under `recipes/beta/spade/` until complete, reproducible learning results are published. The tests validate the experience section, the processor's grouping and its generations against stand ins; they do not establish learning performance. See [issue #482](https://github.com/Human-Agent-Society/reef/issues/482) and [issue #498](https://github.com/Human-Agent-Society/reef/issues/498) for the pieces, [issue #447](https://github.com/Human-Agent-Society/reef/issues/447) for the pipeline they belong to, and [issue #422](https://github.com/Human-Agent-Society/reef/issues/422) for this classification.

[SPADE](https://arxiv.org/abs/2608.19197), self play in adaptive synthetic executable environments, as a Reef recipe: one policy plays an Environment Designer that writes executable environments and a Reasoning Agent that learns in them. Reef knows one task format, Harbor, and writes, checks and plays Harbor tasks in `reef.record2dataset`: the task contract as a prompt, the served model asked through Reef (every proposal a record with a receipt), the reply held to the authoring rules and written with `reef.core.tasks`, Harbor's oracle and nop agents on the result, and the task player for the episodes. That runs as the generator service `reef serve` starts beside the HTTP service. What SPADE adds is the method, and it lives here.

- Paper: [arXiv:2608.19197](https://arxiv.org/abs/2608.19197)
- Reference code: [spade-rl/spade](https://github.com/spade-rl/spade)

## Layout

```text
beta/spade/
  generation.py   the experience section (results sorted by regret into the frontier, the mastered and the out of reach), a generation's records
  processor.py    the reported half: episodes grouped by task; the task generation half: the Designer's generations, run on a worker
  designer_processor.py   the Designer's reports grouped by generation; skills compare through the group id
  objective.py    group relative advantages per group, on Tinker's importance sampling loss; both roles use it
  recipe.py       the two recipes: the Reasoning Agent's on its episodes, the Designer's on its regret
  examples/tinker/serve.yaml            a Tinker deployment with the generator service
  examples/designer-tinker/serve.yaml   a Tinker deployment of the Designer recipe, on its own port
  harness.py      the Designer's prompt as a harness tree, rewritten from the regret reports
  examples/designer-harness/serve.yaml   the Designer's harness evolution service, on its own port
```

## The processor

`SpadeProcessor` implements two of Reef's processor contracts at once.

As a reported feedback processor it takes the task player's reports: every plain episode names its task under `metadata.task`, the episodes of one task form a group, a group is complete at `rollouts-per-task` episodes, and a batch holds `tasks-per-step` complete groups. `SpadeObjective` centers and scales each episode's reward within its task group (a group with one reward everywhere gives 0), and Tinker's built in `importance_sampling` loss puts that advantage on every response token, so the recipe runs on the `tinker` backend today and fails at selection on Slime, which has no loss family of that name. The Designer's own reports (`metadata.role: designer`) and the hint arm's episodes (`arm: hint`) share the scenario; the processor releases them unassembled and trains on the plain arm alone.

As a task generation processor (`reef.train.processors.TaskGenerationProcessor`),
it runs one generation on a private worker while the trainer handles batches.
A generation contains `count` proposals over the configured `skills`:

1. Ask the Designer through the generator's `generate` operation. Each prompt
   includes the previous generation's experience.
2. Write each reply as a Harbor task and run `validate`. The oracle must score 1
   and the nop agent must score below 1. Duplicate task names are refused, except
   for names from an interrupted attempt of the same generation.
3. Play each accepted task `rollouts-per-task` times without a hint for training,
   and `hint-plays` times with `solution/hint.txt` appended for measurement.
4. Report the result against the Designer's receipt. A measured task's score is
   its raw regret: mean hint reward minus mean plain reward. A refused proposal
   gets `REFUSAL_SCORE` (-1.0). Regret can be negative and can equal or fall below
   the refusal score; the refusal score is not a strict lower bound.

The report metadata includes the generation and its size (`proposals`). Feedback
contains the task's measurement and `round.previous`, the preceding generation's
mean regret and counts of measured and refused tasks.

The plain mean classifies a task as mastered (above 0.9), out of reach (below
0.1), or frontier. The next prompt prioritizes frontier tasks by highest regret.
`manifest-<generation>.json` under the tasks root groups tasks by Designer record
ID. `state-dir/generation-<generation>.json` stores proposals, refusals, and
measurements so a restart can resume with the next generation and last experience.

The first generation starts when the processor first looks for a batch; the next once `batches-per-generation` batches were acknowledged since the previous one started (its episodes train while it runs), so the Designer writes for the policy that trains now; a generation that measured no task is followed at once; `generations` caps them. `GET /reef/status` shows the generation in flight, the count completed and the last error.

With `report-plays-after-generation` every play is held until the generation lands, and the generation then reports them all at once: the plain plays stamped with the generation as their round and with the round's size, the hint plays beside them. Nothing trains while the generation runs, so the Designer is measured against one Reasoning Agent version. The processor keeps a round as one unit and trains the generation as one batch, ordered by task, whatever `tasks-per-step` says; the batch counter counts one batch per generation, so `batches-per-generation: 1` puts one training step between generations. Off by default: each play reports as it ends and trains in the next batch of complete groups. The generation's report says how many held plays went out.

## Run it

```bash
export REEF_TOKEN=reef-local REEF_SPADE_STATE_DIR="$PWD/work/spade"   # and TINKER_API_KEY
reef serve -c recipes/beta/spade/examples/tinker/serve.yaml
```

In another terminal, create the scenario to start generation:

```bash
export REEF_TOKEN=reef-local
curl -s -X POST -H "Authorization: Bearer $REEF_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "spade"}' http://127.0.0.1:8900/reef/scenarios
```

The scenario is what the processor belongs to, and `reef serve` creates none on its own: the `POST /reef/scenarios` above (or the first model call that names the scenario) brings it into being, and generation 0 starts on the processor's first look for a batch after that.

The deployment's `generator` section makes `reef serve` start the generator service before the HTTP service and hand its address to the recipe as `${endpoints.generator}`. The generator runs under Reef's interpreter; its host needs Docker and the `harbor` command line, which the Reef service itself does not. `execution: {generator: ray}` places it elsewhere. `generator.designer-url` and `generator.designer-model` point the Designer at another service (a strong model on OpenRouter while the Reasoning Agent is the deployment under training); by default both roles are the served model. `generator.designer-options` adds fields to the Designer's chat request: a model that thinks for thousands of tokens before writing an environment runs past the service's inference deadline, and `{"reasoning_effort": "none"}` keeps it to the reply. See [the generator section](../../../docs/reference/configuration.rst) for every key.

With `generations: 0` the processor generates nothing and trains on whatever the task player reports, which is how to train on tasks written elsewhere:

```bash
python -m reef.harness.client.tasks --reef-url http://127.0.0.1:8900 --scenario spade --model Qwen/Qwen3-8B \
  --manifest tasks/manifest-00000.json --tasks-root tasks --side train --work-dir work/play --label arm=plain
```

Thinking stays off because a thinking model's episode never assembles into one sample: the agent's history carries earlier turns without their thinking, so the second turn's prompt no longer extends the first turn's tokens. With thinking off, Qwen3's generation prompt still ends with an empty think block that the history drops; `scaffold-tolerance` lets the assembly realign those masked tokens. A tasks root under a path Docker shares with the host (on macOS, under the home directory) is required, or the verifier's reward file never reaches the host.

With immediate play reporting, a generation can run for hours while weights reload
after each step, so one task group can include multiple weight versions.
`max-staleness` bounds that lag; `report-plays-after-generation` selects the held
reporting mode described above.

## Choose how the Designer learns

The Reasoning Agent runs on port 8900 with
[`examples/tinker/serve.yaml`](examples/tinker/serve.yaml). To train the Designer,
run a second service on port 8901 using **one** of these alternatives:

| Designer update | Deployment | What changes |
| --- | --- | --- |
| Weights | [designer-tinker](examples/designer-tinker/serve.yaml) | A separate Designer LoRA, trained from proposal regret. |
| Prompt | [designer-harness](examples/designer-harness/serve.yaml) | The `designer-system` and `designer-rules` skill entries. |

Both paths report Designer proposals to the second service. They use separate
policies for the Designer and Reasoning Agent; they do not implement the paper's
shared-weight self play.

## Train the Designer on its regret

Before starting the Reasoning Agent stack, add these fields to its existing
`generator` section in `examples/tinker/serve.yaml`. Keep `tasks-root` and the
other generator settings already in that file:

```yaml
generator:
  designer-url: http://127.0.0.1:8901
  designer-token: ${REEF_TOKEN}
  designer-scenario: designer
  designer-model: Qwen/Qwen3-8B
```

Start the Designer in a separate terminal with `TINKER_API_KEY` set:

```bash
export REEF_TOKEN=reef-local
export REEF_SPADE_DESIGNER_STATE_DIR="$PWD/work/spade-designer"
reef serve -c recipes/beta/spade/examples/designer-tinker/serve.yaml
```

Then start the Reasoning Agent stack and create its scenario as in **Run it**.
Use the same `REEF_TOKEN` in both terminals. The Designer deployment permits
implicit scenario creation: the generator's first call creates `designer`.
`designer-model` must name the model served by that deployment.

Each proposal creates one inference record on the Designer service. Its report
carries raw regret, generation, and `proposals`. `SpadeDesignerProcessor` waits
for all reports in a generation, including refusals, before completing the group.
`SpadeObjective` compares scores within that generation and, when supplied, the
skill. Equal-score groups produce no learning signal; a generation whose
proposals all scored alike is skipped.

`generations-per-step` sets how many complete generations one training step
consumes. Query the Designer service to inspect buffered groups and report counts:

```bash
curl -fsS -H "Authorization: Bearer $REEF_TOKEN" http://127.0.0.1:8901/reef/status
```

A complete generation should lead to a training step or an explicit skip. A skip
because every proposal was refused verifies batching, not successful learning.

## Evolve the Designer's prompt

This alternative uses `SpadeDesignerHarnessRecipe` to evolve two texts: the system
turn and the rules block (`reef.record2dataset.designer.DesignerPrompt`). Its tree
contains the `designer-system` and `designer-rules` skill entries.

### Connect and start the services

In the Reasoning Agent deployment's existing `generator` section, add:

```yaml
generator:
  designer-url: http://127.0.0.1:8901
  designer-token: ${REEF_TOKEN}
  designer-scenario: designer
  designer-prompt: harness
```

Set the Designer service's `recipe.config.batch-size` to the Reasoning Agent
recipe's `count`. This lets one prompt-update step consume one generation of
proposal reports. The shipped example uses 8 for both.

Start the Designer with an upstream endpoint and its credentials:

```bash
export REEF_TOKEN=reef-local
export REEF_SPADE_DESIGNER_STATE_DIR="$PWD/work/designer"
export REEF_UPSTREAM_URL=https://openrouter.ai/api
export REEF_UPSTREAM_MODEL=openai/gpt-5
export REEF_UPSTREAM_API_KEY=...
reef serve -c recipes/beta/spade/examples/designer-harness/serve.yaml
```

Start the Reasoning Agent stack in another terminal and create its scenario as
in **Run it**. Use the same `REEF_TOKEN` in both terminals. Every Designer call
and its regret report now belongs to `designer` on port 8901.

### Follow one generation

1. The generator pulls `native/tree.json` from `GET /reef/harness` once per
   generation and uses its two prompt entries. If the scenario has no tree yet,
   generation uses the fixed prompt.
2. After the generation reports its proposals, `propose_prompt` shows the model
   their regret, feedback, and instruction excerpts alongside the current texts.
   Excerpts are fenced as data. The reply is a JSON array of skill entries;
   only changed texts become `update` mutations.
3. `ReportedRegretSelection` publishes every rewrite without running evaluation
   episodes. The next generation's regret measures the effect of that rewrite.

Inspect the current prompt release with:

```bash
curl -fsS -H "Authorization: Bearer $REEF_TOKEN" \
  -H "x-reef-scenario: designer" http://127.0.0.1:8901/reef/harness
```

Compare its release ID and prompt entries across generations. Use
`state-dir/generation-<generation>.json` on the training side for mean regret and
refusals, and the Designer's `evolution.step-record-dir` for rewrite results.
A new release confirms publication; it does not establish an improvement.

### Wait for an update and interpret failures

Before the first proposal of the next generation, the generator waits for a
Designer version different from the one observed when the previous generation's
reports began. A harness Designer is identified by release; a weight Designer
by runtime load ID and scenario step. A creation release that appeared before
the reports does not satisfy the wait.

`generator.designer-poll-s` controls polling and `generator.designer-wait-s`
bounds the wait. On timeout, generation proceeds with a warning. A deployment
with neither version signal is not waited on.

There is no evaluation step that blocks a worse prompt rewrite. A rewrite that
omits `{turn_limit}` or a task-contract rule may cause refusals in the next
generation; that feedback is available to the following rewrite. Review the
regret and refusal trend rather than treating every publication as an improvement.
