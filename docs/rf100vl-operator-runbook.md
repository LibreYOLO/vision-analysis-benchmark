# RF100-VL operator runbook (human-flown)

You, a browser, and two terminals. No agents anywhere in this document: every
command is typed and every check is read by a person. Expected totals for a
yolov9t campaign on one 8x RTX 5090 on-demand box: about 5 to 7 hours wall
clock, about 13 to 18 dollars.

The mental model in five lines: you rent 8 GPUs by the hour; you put the code
and the dataset on them; ONE command fine-tunes one small model per dataset
(100 of them, 8 at a time), evaluates every one on its test split, and writes
a report; you copy the results off; you destroy the box. Everything on the box
writes plain status files; the dashboard is just a viewer over those files.

## 0. Rehearse locally first (10 minutes, free)

Type the exact commands you will later type on the box, on your own GPU, with
2-epoch throwaway training. This is the whole point: your hands learn the
flow while mistakes are free.

```bash
cd C:/rf100vl
venv/Scripts/python -m va_bench.cli rf100vl-preflight --model yolov9t \
  --data-dir data --weights-root rehearsal-ckpt
venv/Scripts/python -m va_bench.cli rf100vl-campaign --model yolov9t \
  --data-dir data --weights-root rehearsal-ckpt --gpus 0 \
  --limit-datasets 1 --smoke-epochs 2
venv/Scripts/python -m va_bench.cli rf100vl-dash --state-root rehearsal-ckpt/.state
```

Read the preflight lines. Watch the heartbeat lines appear. Open the
dashboard at http://127.0.0.1:8877 and click the dataset. Read the report it
prints at the end and notice it says NOT submittable (a 2-epoch smoke never
is). Delete `rehearsal-ckpt` afterwards. That is the entire skill set.

## 1. Rent the box (browser, 5 minutes)

Go to https://cloud.vast.ai (log in), Search tab:

- Filters: GPU = RTX 5090, GPUs = 8x, Verified, Disk >= 300 GB.
- Instance configuration, image:
  **`vastai/pytorch:2.11.0-cu128-cuda-12.9-mini-py312-2026-06-15`**
  Disk slider: 300 GB.

  This exact tag, for two reasons learned the expensive way. It is one of
  Vast's OWN images, so key injection and sshd take their well-travelled
  provisioning path; a plain `pytorch/pytorch` image works on many hosts but
  on others sshd rejects your key with `bad ownership or modes for file
  /root/.ssh/authorized_keys`, and there is no way in to fix it. And it is
  built with cu128, so it has sm_120 kernels: Vast's default `vastai/pytorch`
  tag raises `no kernel image is available for execution on the device` on a
  5090.

  **The interpreter is `/venv/main/bin/python`, not `python`.** Vast images
  keep their environment in a venv that a non-interactive `ssh host command`
  does not have on PATH. Install with `/venv/main/bin/python -m pip install`,
  and the console scripts land in `/venv/main/bin/`.
- Pricing: On-Demand for your first flights (interruptible is cheaper but
  adds a failure mode you do not need while learning).

**If the console rejects the rental with `invalid_args`, do not fight the
form.** Its template editor has been observed silently reverting edits on
save, cloning a new template per attempt, and preserving a leading space in
the image path (` pytorch/pytorch`), which is not a real image name. Create
the template through the API instead, once:

```bash
vastai create template --name LIBREYOLO-RF100VL \
  --image pytorch/pytorch --image_tag 2.11.0-cuda12.8-cudnn9-runtime \
  --ssh --direct --disk_space 200
```

or skip templates entirely and rent from the CLI, which ignores them:

```bash
vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime \
  --disk 200 --ssh --direct --cancel-unavail --label rf100vl
```

`--cancel-unavail` matters: without it, a taken offer yields a *stopped*
instance that bills its disk while doing nothing. Never pass `--env` port
blocks or `--onstart-cmd entrypoint.sh` with this image; those belong to
Vast's own `vastai/pytorch` image and fail here.

THE PRICE IS ON THE OFFER CARD. That $/hr times the hours until you destroy
is your bill, plus cents of storage and bandwidth. Expect around $2.2 to
$2.8/hr for a good 8x5090. Click RENT. The Instances tab now shows your box
with a live status and its $/hr; it is billing from this moment.

## 1b. Watch the pull for 90 seconds (free, and it catches the commonest dud)

The box spends its first minutes pulling a 6.44 GB image. That pull is also
the first honest test of the host's route to the outside world, and our
workload is unusually dependent on that route: 6.44 GB of image, ~3 GB of pip,
and 43 GB of dataset before a single training step. Watch it:

```bash
vastai show instances-v1 --raw    # read actual_status and status_msg
```

- **Healthy**: distinct layer IDs advancing, `Pull complete` lines
  accumulating, `running` within about 3 to 5 minutes.
- **Sick**: the same three to five layer IDs reappearing with
  `Retrying in N seconds`, no net progress. Give it 90 seconds. If it is still
  cycling the same layers, destroy it.

This gate matters because it fires *before* ssh exists. A box wedged in the
pull never reaches `running`, so section 2b's acceptance test can never run on
it; without this gate the failure is invisible except as money.

**When you replace a dud, check you are actually changing networks.** Offers
carry a `public_ipaddr`, and distinct host accounts in distinct advertised
cities routinely share one egress IP. Measured on 2026-07-31: offer 40871485
("Oregon", host 572017) and offer 45260751 ("California", host 557452) both
egress from `13.56.204.87`, and both wedged identically in the pull, for 7 and
10 minutes, at a cost of about $0.95. A host on a different IP pulled the same
image cleanly and was `running` in 3.4 minutes. Renting "a different host" on
the same egress is not a retry, it is the same experiment.

```bash
# distinct-egress check before you rent the replacement
vastai search offers 'num_gpus=8 rentable=true verified=true' -o dph_total --raw \
  | python -c "import json,sys; [print(o['id'], o['host_id'], o.get('public_ipaddr'), o.get('geolocation')) for o in json.load(sys.stdin)[:15]]"
```

Two scores that do NOT predict this: `reliability2` (both bad boxes scored
0.986 and 0.994; it measures uptime and contract completion, not egress) and
advertised `inet_down` (a bandwidth claim, not a reachability one). The
plausible mechanism is Docker Hub's rate limit, which is 100 requests per hour
keyed on the *source IP* (`docker-ratelimit-source` in the response headers),
so every box behind a shared NAT draws on one bucket; packet loss to the CDN
looks the same from outside. The remedy is the same either way: change egress.

Cheap inbound transfer is worth filtering on for a second reason. Staging
43 GB costs $1.68 at $0.039/GB and $0.11 at $0.0026/GB, and the cheap-transfer
hosts tend to be real colo rather than resold cloud.

## 2. Connect (2 minutes)

Instances tab, the ">_" (Connect) button shows the exact line, shaped like:

```bash
ssh -p <PORT> root@<HOST-IP>
```

`Permission denied (publickey)` on the first attempt is normal rather than
broken. Your public key has to be on the account AND on the instance, and
propagation takes a few seconds:

```bash
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"   # once per account
vastai attach ssh <INSTANCE_ID> "$(cat ~/.ssh/id_ed25519.pub)"
```

Then retry, naming the key explicitly (`ssh -i ~/.ssh/id_ed25519 ...`), and
try the direct address (`public_ipaddr` with `direct_port_start`, from
`vastai show instances-v1`) if the proxy host keeps refusing.

It works from PowerShell or Git Bash. First contact, look around:

```bash
nvidia-smi                 # expect 8x RTX 5090, 32GB each, idle near 0%
df -h /                    # expect 300G, mostly free
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
                           # expect 2.11.0+cu128 and sm_120 in the list
```

## 2b. Accept or reject the box (60 seconds, do this FIRST)

Rented hosts vary enormously and the marketplace does not grade the things
that matter to us. Run the acceptance test before installing anything:

```bash
scp -P <PORT> deploy/vast/accept-box.sh root@<HOST>:/root/
ssh -p <PORT> root@<HOST> bash /root/accept-box.sh
```

or inline, if you would rather not copy a file:

```bash
echo "hf: $(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
  https://huggingface.co/api/datasets/LibreYOLO/rf100-vl)"
nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
/venv/main/bin/python -c "import torch; print('sm120:', 'sm_120' in torch.cuda.get_arch_list())"
```

Want `hf: 200`, your GPU count, and `sm120: True`. Anything else: destroy the
box immediately and rent another. A dud caught here costs about five cents;
the same dud caught after staging 43 GB costs twenty minutes and a dollar.

**Expect to burn two or three hosts.** In one evening we hit, on separate
machines: an offer that would not schedule, a broken NVIDIA container runtime
(`OCI runtime create failed`), two hosts whose networks could not reach
huggingface.co (one IPv6-only with no IPv6 egress, one blocked outright while
PyPI and GitHub worked), a host that wrote `authorized_keys` with bad modes,
and one that could not pull from Docker Hub at all. None of these are
detectable from the offer card. Budget roughly a dollar for the search.

The most dangerous of those is the network one, because `snapshot_download`
does not error: it hangs silently, forever, while the meter runs.

## 3. Install (paste block, ~3 minutes)

```bash
P=/venv/main/bin/python                # the venv interpreter, see section 1
apt-get update -qq && apt-get install -y -qq git tmux rsync \
  libgl1 libglib2.0-0 libxcb1 libxrender1 libxext6 libsm6

$P -m pip install -q \
  "libreyolo[rfdetr] @ git+https://github.com/LibreYOLO/libreyolo@19b9321ab11450f9d9569ca059e2346267a51aca"
$P -m pip install -q \
  "git+https://github.com/LibreYOLO/vision-analysis-benchmark.git@rf100vl-harness" \
  pycocotools huggingface_hub psutil pyyaml nvidia-ml-py

# Prove the install AND that nothing downgraded torch out of cu128.
$P -c "import torch, libreyolo; print(torch.__version__, 'sm120', 'sm_120' in torch.cuda.get_arch_list(), 'libreyolo', libreyolo.__version__)"
$P -c "import pynvml; pynvml.nvmlInit(); print('nvml OK,', pynvml.nvmlDeviceGetCount(), 'devices')"
/venv/main/bin/va-bench --help | head -3
```

Expect `2.11.0+cu128 sm120 True libreyolo 1.4.0` and `nvml OK, 8 devices`.

`nvidia-ml-py` is what makes the per-GPU telemetry work. Leave it out and the
campaign still runs, but it prints one line, `pynvml unavailable, no GPU
telemetry`, and then records nothing about utilisation or power. That is easy
to miss in a wall of preflight PASSes, and it costs you the answer to the
question the run was partly meant to settle: whether the GPUs were actually
busy. Install `nvidia-ml-py`, not `pynvml`; the latter is a deprecated shim
that works but warns on every process start. If torch lost `+cu128`, a
dependency pulled a different wheel: reinstall it from the cu128 index before
going further, because training would silently fall back or fail to launch
kernels. (`PIP_BREAK_SYSTEM_PACKAGES=1` is only needed on plain Debian-based
images, not on the Vast image, which installs into its own venv.)

The libreyolo commit above is the validated campaign pin. Bump it
deliberately after testing, never casually.

## 4. Stage the dataset (~10-15 minutes at datacenter speed)

```bash
cd /root
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('LibreYOLO/rf100-vl', repo_type='dataset', \
                    local_dir='rf100-vl', max_workers=8)"
cd rf100-vl && for f in *.tar; do tar xf "$f" && rm "$f"; done && cd /root
rm -f ./-grccs.tar                     # see below
ls rf100-vl | wc -l                    # expect ~104 (100 datasets + metadata)
```

Two things this step reliably teaches:

- **Run the download and the extraction as separate commands.** Pasting a
  multi-line block queues the later lines, so a Ctrl-C during the download
  leaves the extract loop running against a half-finished directory. The
  download itself resumes fine: re-run the same snapshot_download.
- **One dataset is named `-grccs`.** The `rm "$f"` in that loop reads
  `-grccs.tar` as command-line flags and fails (tar itself is fine, so the
  data is extracted). Delete it with a path prefix: `rm ./-grccs.tar`. The
  same trap applies to the harness flags, hence `--datasets=-grccs`.

## 5. Preflight (30 seconds, reads like a checklist)

```bash
va-bench rf100vl-preflight --model yolov9t \
  --data-dir /root/rf100-vl --weights-root /root/rf100vl-weights
```

Seven lines, every one must say PASS (libreyolo capabilities, data vs its
version lock, split files, recipe hash, torch-vs-GPU architecture, disk,
writability). A FAIL here costs you box-minutes, not GPU-hours. Fix or
destroy; never proceed past a FAIL.

## 6. tmux, then a smoke (10 minutes)

The run must survive your wifi. Vast already logs you into a tmux session
called `ssh_tmux`, so you are protected from the first keystroke and a new
`ssh` drops you back into the same session with your jobs still running. Do
not nest a second session inside it. The shortcuts worth knowing:

```
Ctrl-b d      detach (everything keeps running); ssh back in to return
Ctrl-b c      new window, e.g. for nvidia-smi or the dashboard
Ctrl-b n / p  next / previous window
Ctrl-b [      scroll mode (PgUp/PgDn, q quits)
tmux set -g mouse on    # once, if you prefer the scroll wheel
```

Inside tmux, a two-dataset, two-epoch shakeout of the full pipeline:

```bash
va-bench rf100vl-campaign --model yolov9t \
  --data-dir /root/rf100-vl --weights-root /root/smoke-weights \
  --gpus 0,1 --limit-datasets 2 --smoke-epochs 2
```

Expect: preflight passes again, a progress heartbeat line about every 60
seconds, both datasets done in minutes, and a report that says NOT
submittable (correct: it is a smoke). Use a separate weights root for smokes,
as here; the orchestrator quarantines smoke leftovers, but separate roots
keep the campaign history clean.

## 7. The real run (one command, then hours)

Still inside tmux:

```bash
va-bench rf100vl-campaign --model yolov9t \
  --data-dir /root/rf100-vl --weights-root /root/rf100vl-weights \
  --gpus 0,1,2,3,4,5,6,7
```

Detach (Ctrl-b d). If ANYTHING dies, up to and including the box, re-running
this exact command resumes: finished datasets are skipped, the interrupted
one restarts from its last epoch checkpoint.

The napkin math you should carry in your head: 100 datasets / 8 GPUs is 12 or
13 per GPU, at roughly 20 to 40 minutes each, so 4 to 6 hours; times the $/hr
on your instance card is the bill.

## 8. Watch it (pick any or all)

**Dashboard in your browser.** On the box (tmux window 2: Ctrl-b c):

```bash
va-bench rf100vl-dash --state-root /root/rf100vl-weights/.state
```

On your laptop, forward the port through SSH (this is the answer to "can
vast forward that port": you pull it through the ssh connection; nothing is
ever exposed publicly):

```bash
ssh -p <PORT> -L 8877:127.0.0.1:8877 root@<HOST-IP> -N
```

Open http://127.0.0.1:8877. Per-GPU lanes, the 100-dataset grid, click any
dataset for its live loss and mAP curves and log tail.

**GPUs, raw.** In another tmux window: `watch -n 5 nvidia-smi`. Healthy
training shows every GPU cycling roughly 60 to 100 percent utilization and
roughly 200 to 400 W. One GPU at 0 percent for over 10 minutes while the
dashboard shows it owning a running dataset means something is wrong.

**Money.** The Instances tab card shows $/hr; the Billing page shows actual
charges accruing. Sanity-check it against hours-elapsed times the card price
once or twice during the run.

**The heartbeat.** `tmux attach -t bench` any time: one line per minute,
done/running/pending/failed plus per-GPU epoch, best mAP, and ETA.

Healthy looks like: heartbeat counts advancing, lanes climbing epochs, curves
rising, pending draining. Red flags: a lane marked stale in the dashboard, a
0 percent GPU with a running lane, disk above 90 percent, failed count
climbing, or a card price that does not match what you expected.

## 9. Finish: read, pull, destroy

The campaign ends by printing the markdown report (domain means, ok/100,
train cost, weakest datasets) and writing it next to the submission JSON in
`/root/results_rf100vl/`. Read it on the box, then pull everything you want
to keep to your laptop:

```bash
# from your laptop
scp -P <PORT> -r root@<HOST-IP>:/root/results_rf100vl ./rf100vl-results
scp -P <PORT> -r root@<HOST-IP>:/root/rf100vl-weights ./rf100vl-weights   # ~3GB, optional
```

Off-box archive to Hugging Face. Set the token up ONCE, before the campaign,
and scope it as narrowly as the job allows:

1. Create the dataset repo by hand at https://huggingface.co/new-dataset
   (owner `LibreYOLO`, name `rf100-vl-results`). Doing this manually means the
   token never needs repo-creation rights, which is the broad permission worth
   avoiding.
2. Settings -> Access Tokens -> New token -> **Fine-grained**. Under
   Repository permissions pick that one repo and tick only
   **Write access to contents/settings**. Nothing at org or user level.
3. `export HF_TOKEN=hf_...` on the box, never in a committed file. Revoke it
   when the campaign ends.

The uploader tolerates exactly this: it attempts `create_repo` but treats a
refusal as fine when the repo already exists and is writable.

```bash
va-bench sync-artifacts --model yolov9t --run-id <today>-yolov9t \
  --weights-root /root/rf100vl-weights --data-dir /root/rf100-vl --tier all
```

Then, in the console Instances tab: DESTROY (the trash icon). Not "stop":
a stopped instance keeps billing its disk forever. Destroy, then confirm the
Instances tab is empty. That is the moment billing ends.

## Aborting at any point

**Stop the run, keep the box:** Ctrl-C in the tmux window. The orchestrator
terminates the trainers, marks the datasets that were mid-flight as pending,
and exits. Re-running the identical command resumes: finished datasets are
skipped, interrupted ones continue from their last epoch checkpoint.

**Stop everything:** destroy the box. Always safe, always ends the spend. You
lose whatever was not pulled or synced (at worst, partial training that would
re-run next time); you keep everything you copied off. There is no state
anywhere except that box, your laptop, and whatever you pushed to HF.

## Answering questions about the benchmark (the honest cheat sheet)

- Protocol: per dataset, fine-tune the COCO-pretrained checkpoint 100 epochs
  (fixed recipe, effective batch 16, seed 0), select the best-on-valid EMA
  checkpoint, score it on the test split with pycocotools at maxDets 500;
  headline = unweighted mean AP50:95 over the 100 datasets.
- Why trust the number: the submission JSON records the harness and libreyolo
  commits, recipe hash, dataset version lock, and per-dataset raw predictions;
  anyone can rescore those predictions with stock pycocotools and get the
  same mean to four decimals, no GPU needed (`va-bench rescore`).
- What it cost: the Vast invoice, plus the report's train-cost section
  (median minutes per dataset, total GPU-hours).
- What failed: the report's completion line (ok/100) and
  `<weights-root>/.state/<model>/failures.json`, which records every failure
  with a traceback tail.
