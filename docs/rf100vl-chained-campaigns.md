# Chained campaigns: running several models in one night

Design note, not yet built. Written 2026-07-31 while the single-model path was
being proven, so the constraints below come from a real campaign rather than
from imagination.

## What it is for

Run a family in one booking: `yolov9t, yolov9s, yolov9m, yolov9c`. Or run
across families: `yolov9t, yolox-nano`. The expensive parts of a campaign are
renting the box, staging 43 GB, and installing. Those are paid once and then
amortised over every model in the chain, so the second model is much cheaper
than the first.

## The one rule that matters

**Sequential with a hard gate, never a joint budget.**

A chain that runs models one after another, each only starting when the
previous one genuinely completed, degrades gracefully: you wake up to N
complete results and one partial. A chain that splits the window between
models degrades catastrophically: you wake up to several incomplete runs, none
of which is publishable. The asymmetry is the whole design.

Corollary: **order cheapest-first**. Put the model most likely to finish at
the front, so the partial one is always the most expensive.

Today this is already achievable by hand and is the recommended workaround:

```bash
va-bench rf100vl-campaign --model yolov9t --weights-root /root/w-t --jobs-per-gpu 3 ... \
&& va-bench rf100vl-campaign --model yolov9s --weights-root /root/w-s --jobs-per-gpu 2 ...
```

Separate weights roots are mandatory, or `stats.json` files collide.

## What a real implementation must add

**Per-model lane count.** Packing depth is not a constant. One training of
yolov9t is 6.4 GB and three fit a 24 GB card, but a larger model needs fewer
lanes, and there is a second ceiling above VRAM: a training occupying X% of
GPU time saturates the card at roughly 1/X lanes, past which extra lanes add
no throughput and stretch the tail job. So a chain needs either a declared
`jobs_per_gpu` per model or a short probe that measures VRAM and GPU-time
share on one dataset and derives it.

**A time budget with graceful surrender.** Given "stop by 07:00", the chain
should decline to START a model it cannot finish, rather than beginning one
and being killed mid-dataset. Partial datasets are resumable, so this is a
preference rather than a correctness issue, but a model that never started is
tidier than one abandoned at 40%.

**Idempotent resume across the whole chain.** Re-running the same chain
command should skip completed models entirely and resume the interrupted one.
Per-dataset state already works this way; the chain needs the same at model
granularity, in a `chain.json` beside the per-model state.

**Failure policy, explicitly chosen.** `--on-failure stop` (default) versus
`--on-failure continue`. A model failing because its weights are missing
should not consume the night; a model failing on three datasets probably
should not block the next model.

**One provenance manifest per model.** Each model is its own run id with its
own commits and hashes. Already true today, and the derived run id keeps them
from colliding.

**Teardown once, at the end of the chain**, not after each model. This is the
same missing piece that blocks unattended single-model runs: something has to
destroy the box when the last model finishes. Vast issues an
`instance_api_key` at creation that works from inside the instance.

## Cross-family notes

Recipes are per family and resolved by `recipe_path_for_family`, so a chain
spanning families needs no new machinery, but each family's recipe must exist
and satisfy the protocol skeleton. Mixing families in one chain is where a
per-model probe earns its keep, since VRAM and GPU-time share differ far more
across families than across sizes within one.

## Suggested surface

```bash
va-bench rf100vl-chain \
  --models yolov9t:3 yolov9s:2 yolov9m:1 \
  --data-dir /root/rf100-vl --roots-under /root/campaigns \
  --sync-repo LibreYOLO/rf100-vl-results \
  --stop-by 2026-08-01T07:00:00Z --on-failure stop
```

`model:lanes` keeps the packing decision explicit and visible in the command
that produced the results, which matters when someone later asks why two runs
of the same model differ.
