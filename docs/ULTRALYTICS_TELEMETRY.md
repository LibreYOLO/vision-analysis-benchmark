# Ultralytics telemetry audit (v8.4.60) and the isolation recipe

Audited from source at tag v8.4.60. All `file:line` references are into the
`ultralytics` package source at that version. This doc exists because the
va-bench driver must guarantee that benchmarking Ultralytics models sends no
telemetry from the benchmark machine. Facts below are verifiable in their
source; nothing here is speculation.

## Outbound channels found

| # | Channel | Endpoint | Fires on | Gate (all must hold) |
|---|---|---|---|---|
| 1 | GA4 usage events (`utils/events.py`) | `www.google-analytics.com/mp/collect?measurement_id=G-X8NCJYTQXM&...` (events.py:46) | every train/val/**predict**/export via HUB callbacks (callbacks/hub.py:77-94) | `SETTINGS["sync"]` AND rank 0/-1 AND not tests AND `ONLINE` AND pip-install-or-their-git (events.py:64-70) |
| 2 | Sentry crash reports (`utils/__init__.py:1159-1224`) | `o4504521589325824.ingest.us.sentry.io` (utils/__init__.py:1215) | unhandled exceptions, `yolo` CLI only | `sync` AND `yolo` argv AND `ONLINE` AND pip pkg AND `sentry_sdk` importable |
| 3 | HUB (`hub/utils.py`, `hub/session.py`) | `api.ultralytics.com` | training heartbeats/uploads | `SETTINGS["hub"]` AND non-empty `api_key` |
| 4 | Platform (`utils/callbacks/platform.py`) | `platform.ultralytics.com` | training console/metrics/checkpoint streaming | `SETTINGS["platform"]` or `ULTRALYTICS_API_KEY` env or `api_key` setting |

Notable payload facts:

- The GA4 `client_id` and Sentry user id are `SETTINGS["uuid"]` =
  **SHA-256 of the machine MAC address** (`hashlib.sha256(str(uuid.getnode()))`,
  utils/__init__.py:1363) — a stable, re-derivable machine identifier. It is
  generated and persisted to `settings.json` on the **first import**, even if
  nothing is ever run.
- GA4 events include CPU model, python version, OS, install type, task, model
  name, device string (events.py:53-96).
- Channel 4's payload is the most identifying (hostname, full argv, git
  commit/branch/message, GPU) but is opt-in via API key.
- The GA4 POST uses **raw `urllib`**, not `requests` (events.py:8,16-23) —
  interception layers that only patch `requests` miss it.
- `import ultralytics` fires a **DNS probe** to `one.one.one.one` /
  `dns.google` at import time (`ONLINE = is_online()`, utils/__init__.py:955)
  unless `YOLO_OFFLINE=true` (checked first, :823). No payload, but it is
  outbound network activity.

## Defaults

`sync` defaults to **True** (utils/__init__.py:1364): GA4 usage telemetry is
on out of the box and fires on ordinary `predict`. `api_key` defaults empty,
so channels 3 and 4 are dead by default.

## The trap: partial settings files self-heal to defaults

`_validate_settings` (utils/__init__.py:1395-1406) resets `settings.json` to
defaults — **including `sync: true`** — unless the file contains the exact
full key set with correct types AND `settings_version == "0.0.6"`. A
hand-written file with only `{"sync": false}` silently re-enables telemetry.
The kill file must be the complete 19-key schema.

## Settings file location (Windows)

Default: `C:\Users\<user>\AppData\Roaming\Ultralytics\settings.json`
(utils/__init__.py:923-924,966-967). If `YOLO_CONFIG_DIR` is set, the path is
`<YOLO_CONFIG_DIR>/Ultralytics/settings.json` — note the appended
`Ultralytics` subdirectory (utils/__init__.py:919-920).

## Isolation recipe (what the va-bench driver implements)

Before the first `import ultralytics` in the driver process:

1. `YOLO_OFFLINE=true` — kills the import-time DNS probe and the `ONLINE`
   gate on channels 1 and 2. Strongest single switch.
2. `YOLO_AUTOINSTALL=false` — no silent `pip install` subprocesses.
3. Ensure `ULTRALYTICS_API_KEY` is **unset** (it would arm channel 4).
4. `YOLO_CONFIG_DIR=<repo>/drivers/ultralytics/config` — the global
   `AppData\Roaming` location is never touched.
5. Pre-write the **complete** settings.json (all 19 keys,
   `settings_version: "0.0.6"`, `sync: false`, `api_key: ""`, and a fixed
   non-MAC-derived `uuid` placeholder) at
   `<YOLO_CONFIG_DIR>/Ultralytics/settings.json`, so the MAC-hash identifier
   is never generated.
6. **Socket guard**: the driver monkeypatches `socket.socket` before import
   and keeps it blocked for the whole run — any attempted network call from
   any library raises immediately and fails the run loudly. Telemetry-off is
   proven at runtime, not assumed. (Weight fetching happens in a separate
   process that never imports ultralytics — see below.)
7. After import, the driver asserts `SETTINGS["sync"] is False` and that the
   active settings file lives under our config dir; it refuses to run
   otherwise, and records both in the output JSON.

## Residual network calls (all avoided)

| Call | Trigger | Our answer |
|---|---|---|
| Weight auto-download from `github.com/ultralytics/assets/releases/download/v8.4.0/<name>` (utils/downloads.py:462-514) | model name not found locally | weights pre-fetched by `fetch_weights.py` with plain urllib — their code never downloads |
| `Arial.ttf` font download (utils/checks.py:323-349) | first predict that draws/saves annotated output | we never draw/save; offline+socket guard would catch it anyway |
| PyPI version check (utils/checks.py:279-318) | `model.train()` only | we don't train; offline kills it |
| AMP check downloading a model (utils/checks.py:847-901) | training on CUDA only | we don't train |
| Auto pip-install of missing deps | missing optional deps | `YOLO_AUTOINSTALL=false` |
