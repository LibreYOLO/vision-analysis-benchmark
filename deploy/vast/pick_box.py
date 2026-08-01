"""Screen live Vast.ai offers for an RF100-VL campaign, and rank the survivors.

Run this at launch time rather than trusting a shortlist someone wrote earlier.

A warning about the search itself, learned by getting it wrong: **vastai
search offers returns a small page by default**, 64 rows in one measurement.
Screening that page and concluding "those offers are gone" is a mistake, and
it produced exactly that false conclusion once. Pass an explicit --limit; the
same query then returned 681 offers, including several that had been declared
delisted minutes earlier.

Every gate below comes from something measured rather than assumed:

* **Throughput.** ~0.72 GPU-hours per dataset on an RTX 4090 at one training
  per GPU, so 100 datasets is ~72 4090-GPU-hours. A box must deliver that
  inside the window, after setup.

* **The makespan floor.** The largest dataset (8791 training images) takes
  ~3.65 h alone on one unpacked 4090 lane, and longer when its lane is shared.
  No amount of GPUs beats it, so a box whose tail job overruns the window is
  rejected however fast it looks in aggregate.

* **VRAM.** One training is 6.4 GB and each additional one on the same card
  adds ~5.65 GB (measured 1 job = 6.4 GB, 3 jobs = 17.7 GB). A 12 GB card
  cannot hold two, which is why cheap low-VRAM boxes fail on capacity rather
  than on price.

* **CPU per lane.** At one training per GPU the GPUs sat at 15.4% utilization
  while power reached only 22% of cap: this workload is dataloader-bound, not
  GPU-bound. Measured later with 3 lanes/GPU: 46 ms GPU vs 507 ms CPU per
  step and 8 cores/lane still ~94% CPU-saturated. Cores per training lane
  therefore predict real throughput better than TFLOPs, and a box that packs
  deeper than its cores allow will not deliver its nominal capacity. Size
  for epoch 1 (cache fill) as well as steady state.

* **Shared egress.** The failure that cost the most money was two offers under
  DIFFERENT host accounts, in DIFFERENT advertised cities, sharing one egress
  IP, both wedged. `reliability2` scored them 0.986 and 0.994 because it
  measures uptime, not whether the box can reach the internet. Counting
  distinct host accounts behind one IP reproduces that signature exactly and
  needs no external lookup.

Read-only. This never rents anything.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys

# Measured constants. Change these only with new measurements.
GPU_HOURS_PER_DATASET_4090 = 0.72
VRAM_FIRST_JOB_GB = 6.4
VRAM_EXTRA_JOB_GB = 5.65
VRAM_USABLE_FRACTION = 0.93
LONGEST_DATASET_EPOCH_SECONDS = 9.07 + 0.0139 * 8791  # 100 epochs of the worst one
PACKING_MULTIPLIER = {1: 1.0, 2: 1.6, 3: 2.0}  # 3 is measured; 2 is interpolated
# Was 3.0; measured campaigns at 8 cores/lane were still ~94% CPU-saturated
# before the post-resize image cache. Keep the floor high until a shakedown
# on the current stack re-measures steady-state need.
MIN_CORES_PER_LANE = 8.0

# Egress IPs that have already cost us money. The shared-host-account heuristic
# does not always catch these: an IP can carry a single host account today and
# still be the cloud NAT that wedged two rentals yesterday. Measured failures
# belong in a list, not in a heuristic.
KNOWN_BAD_EGRESS = {
    # AWS us-west-1 NAT. Two offers under different host accounts, advertised
    # as Oregon and California, both wedged mid-image-pull for 7 and 10 minutes
    # on 2026-07-31 at a cost of about $0.95.
    "13.56.204.87": "wedged two rentals mid-pull (AWS us-west-1 NAT)",
}

# Speed relative to an RTX 4090. Deliberately conservative: the workload is
# dataloader-bound, so newer silicon delivers less than its spec sheet implies.
SPEED_VS_4090 = {
    "RTX 5090": 1.30, "RTX 4090": 1.00, "RTX 4090D": 0.90, "RTX 3090": 0.55,
    "RTX 3090 Ti": 0.60, "RTX A6000": 0.65, "RTX 6000Ada": 1.05, "L40S": 1.00,
    "A100 PCIE": 0.80, "A100 SXM4": 0.90, "H100 PCIE": 1.30, "H100 SXM": 1.50,
    "H200": 1.60, "RTX 5080": 0.85, "RTX 4080": 0.75, "RTX 4080S": 0.78,
}


def jobs_per_gpu(vram_mb: float, cap: int) -> int:
    usable = (vram_mb / 1024.0) * VRAM_USABLE_FRACTION
    if usable < VRAM_FIRST_JOB_GB:
        return 0
    return max(1, min(cap, int(1 + (usable - VRAM_FIRST_JOB_GB) // VRAM_EXTRA_JOB_GB)))


def evaluate(offer: dict, args, host_ids_per_ip, offers_per_ip) -> dict | None:
    name = offer.get("gpu_name", "")
    speed = SPEED_VS_4090.get(name)
    if speed is None:
        return None
    gpus = int(offer.get("num_gpus", 0))
    vram = float(offer.get("gpu_ram", 0))
    cores = float(offer.get("cpu_cores_effective") or 0)

    lanes_per_gpu = jobs_per_gpu(vram, args.max_jobs_per_gpu)
    if lanes_per_gpu == 0:
        return None
    # Do not pack deeper than the CPU can feed: a lane with too few cores does
    # not add throughput, it removes it from its neighbours.
    while lanes_per_gpu > 1 and cores / (lanes_per_gpu * gpus) < MIN_CORES_PER_LANE:
        lanes_per_gpu -= 1

    multiplier = PACKING_MULTIPLIER.get(lanes_per_gpu, PACKING_MULTIPLIER[3])
    equivalents = gpus * speed * multiplier
    work_hours = GPU_HOURS_PER_DATASET_4090 * args.datasets

    staging_hours = (args.stage_gb * 8 * 1000) / max(1.0, offer.get("inet_down", 1)) / 3600
    setup_hours = args.setup_minutes / 60.0 + staging_hours
    window = args.window_hours - setup_hours
    if window <= 0:
        return None

    throughput_hours = work_hours / equivalents
    # The tail job runs on a shared lane, so it is slowed by the same factor
    # that packing buys in aggregate.
    slowdown = lanes_per_gpu / multiplier
    tail_hours = (LONGEST_DATASET_EPOCH_SECONDS * 100 / 3600) / speed * slowdown
    makespan = max(throughput_hours, tail_hours)

    finishes = makespan <= window
    rate = float(offer.get("dph_total", 0))
    disk_hourly = args.disk_gb * float(offer.get("storage_cost", 0)) / 730.0
    billed_hours = makespan + setup_hours if args.self_destruct else args.window_hours
    cost = rate * billed_hours + disk_hourly * billed_hours
    cost += args.stage_gb * float(offer.get("inet_down_cost", 0))

    ip = offer.get("public_ipaddr")
    return {
        "id": offer["id"], "gpu": name, "gpus": gpus, "vram_gb": vram / 1024,
        "lanes_per_gpu": lanes_per_gpu, "concurrent": lanes_per_gpu * gpus,
        "equivalents": equivalents, "required": work_hours / window,
        "slack_pct": 100 * (window - makespan) / window,
        "makespan_h": makespan, "tail_h": tail_hours, "window_h": window,
        "cores_per_lane": cores / max(1, lanes_per_gpu * gpus),
        "rate": rate, "cost": cost, "finishes": finishes,
        "frac": offer.get("gpu_frac"), "rel": offer.get("reliability2", 0),
        "ip": ip, "hosts_on_ip": len(host_ids_per_ip.get(ip, set())),
        "offers_on_ip": offers_per_ip.get(ip, 0),
        "down": offer.get("inet_down", 0), "in_cost": offer.get("inet_down_cost", 0),
        "geo": str(offer.get("geolocation"))[:18],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=int, default=100)
    parser.add_argument("--window-hours", type=float, default=12.0)
    parser.add_argument("--setup-minutes", type=float, default=12.0)
    parser.add_argument("--stage-gb", type=float, default=49.4)
    parser.add_argument("--disk-gb", type=float, default=100.0)
    parser.add_argument("--max-jobs-per-gpu", type=int, default=3,
                        help="3 is the measured ceiling; deeper is extrapolation")
    parser.add_argument("--min-slack-pct", type=float, default=25.0)
    parser.add_argument("--self-destruct", action="store_true",
                        help="Bill only until the run ends (needs the box to stop itself)")
    parser.add_argument("--allow-shared-egress", action="store_true")
    parser.add_argument("--vastai", default="vastai")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--limit", type=int, default=2000,
                        help="Vast paginates; the default page is far too small")
    parser.add_argument("--explain", action="store_true",
                        help="Show why candidates were rejected")
    args = parser.parse_args()

    query = "num_gpus>=2 rentable=true verified=true disk_space>150"
    raw = subprocess.run(
        [args.vastai, "search", "offers", query, "-o", "dph_total",
         "--limit", str(args.limit), "--raw"],
        capture_output=True, text=True, check=True,
    ).stdout
    offers = json.loads(raw)

    # Sibling counts are a LOWER BOUND, not a census. A single query never
    # returns the whole market, and offers that are currently RENTED do not
    # appear at all, so an egress can look solo in one snapshot and shared by
    # five machines in another. The known-bad list exists precisely because
    # this heuristic cannot be trusted to fire on its own.
    host_ids_per_ip: dict[str, set] = collections.defaultdict(set)
    offers_per_ip: collections.Counter = collections.Counter()
    for offer in offers:
        ip = offer.get("public_ipaddr")
        host_ids_per_ip[ip].add(offer.get("host_id"))
        offers_per_ip[ip] += 1

    rows = []
    rejected: collections.Counter = collections.Counter()
    near_misses = []
    for offer in offers:
        row = evaluate(offer, args, host_ids_per_ip, offers_per_ip)
        if row is None:
            rejected["unknown gpu or too little VRAM for one training"] += 1
            continue
        if not row["finishes"]:
            reason = ("tail job overruns the window"
                      if row["tail_h"] > row["window_h"] else "not enough throughput")
            rejected[reason] += 1
            near_misses.append(row)
            continue
        if row["slack_pct"] < args.min_slack_pct:
            rejected[f"less than {args.min_slack_pct:.0f}% slack"] += 1
            near_misses.append(row)
            continue
        if row["ip"] in KNOWN_BAD_EGRESS:
            rejected[f"known-bad egress: {KNOWN_BAD_EGRESS[row['ip']]}"] += 1
            continue
        if not args.allow_shared_egress and row["hosts_on_ip"] > 1:
            rejected["shared egress: several host accounts on one IP"] += 1
            near_misses.append(row)
            continue
        rows.append(row)

    rows.sort(key=lambda r: r["cost"])
    print(f"scanned {len(offers)} offers; {len(rows)} clear every gate")
    print(f"work {GPU_HOURS_PER_DATASET_4090 * args.datasets:.0f} 4090-GPU-h, "
          f"window {args.window_hours}h, billing "
          f"{'to completion' if args.self_destruct else 'the whole window'}\n")
    header = ("offer", "gpu", "lanes", "conc", "equiv", "slack", "span", "tail",
              "cores/lane", "$/hr", "TOTAL", "frac", "rel", "ip(hosts)", "geo")
    print("%-9s %-9s %-5s %-5s %-6s %-6s %-6s %-6s %-10s %-7s %-7s %-5s %-6s %-16s %s" % header)
    for r in rows[: args.top]:
        print("%-9s %dx%-7s %-5d %-5d %-6.1f %-6.0f%% %-6.1f %-6.1f %-10.1f %-7.3f $%-6.2f %-5s %-6.3f %-16s %s" % (
            r["id"], r["gpus"], r["gpu"].replace("RTX ", ""), r["lanes_per_gpu"],
            r["concurrent"], r["equivalents"], r["slack_pct"], r["makespan_h"],
            r["tail_h"], r["cores_per_lane"], r["rate"], r["cost"],
            f"{float(r['frac'] or 0):.2f}", r["rel"],
            f"{r['ip']}({r['hosts_on_ip']})", r["geo"]))
    if args.explain or not rows:
        print()
        print("rejected:")
        for reason, count in rejected.most_common():
            print(f"  {count:4d}  {reason}")
        near_misses.sort(key=lambda r: -r["slack_pct"])
        if near_misses:
            print()
            print("closest misses:")
            for r in near_misses[:5]:
                print("  %-9s %dx%-8s slack %5.0f%% span %.1fh tail %.1fh "
                      "cores/lane %.1f hosts-on-ip %d $%.2f" % (
                          r["id"], r["gpus"], r["gpu"].replace("RTX ", ""),
                          r["slack_pct"], r["makespan_h"], r["tail_h"],
                          r["cores_per_lane"], r["hosts_on_ip"], r["cost"]))
    if not rows:
        print()
        print("Nothing clears the gates. Loosen --min-slack-pct or widen the window.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
