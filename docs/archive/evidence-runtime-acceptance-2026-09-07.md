# Evidence Runtime local acceptance — 2026-09-07

Archived from the RepreGuard README on 2026-09-23. The excerpts below preserve the
recorded results and limitations verbatim, including status wording from that date.
They are historical evidence, not a current environment check or authorization for
new deployment or certification. No acceptance checks were rerun for this archive.

Current project status is maintained only in the
[Evidence V1 tracker](../../../../AIDetector-evidence-research/EVIDENCE_V1_IMPLEMENTATION_TRACKER.md).
Current operating instructions remain in the [RepreGuard README](../../README.md).
Paths in the excerpts retain their original meaning relative to RepreGuard or the
research repository as indicated; the underlying reports remain at those paths.

## Isolated py3langid candidate screen

The 2026-09-07 isolated candidate screen passed its predeclared engineering budgets:
191/192 supported-language documents retained, 24/24 synthetic out-of-scope examples
rejected, and 16/16 formatting variants retained. The one rejection was a Chinese
product listing classified as `wuu`; it also fails the existing ten-sentence rule,
but remains counted as a language-check error. Local CPU P95 was 1.84 ms for normal
inputs and 5.78 ms for repeated 20,000-character timing inputs, with 75.46 MiB added
RSS after loading (NumPy already imported). These are small-sample local results,
not production compatibility, concurrency or accuracy guarantees. The research
repository records the protocol and results in `outputs/py3langid_screen_20260907.json`.
Only the isolated research cache received an installation; the existing lab/service
environment has not been changed. Tests can use that existing cache through their
process-local Python path; deployment must provide the declared dependency.

## Local real-model acceptance (batch 3)

2026-09-07 result (`evidence-runtime-20260907-batch3-01.json`, SHA-256
`72211f3f8ec3624d63acb762e1d53f4d0a3bc61b7edf6ea4af579091fc0b5f86`):

| Measurement | Evidence off | Evidence on |
|---|---:|---:|
| Startup, including imports/model load | 15.94 s | 18.27 s |
| Main P95, Evidence idle | 14.83 ms | 19.51 ms |
| Main P95, concurrent Evidence | — | 31.96 ms |
| Evidence P95, concurrent 20k-character input | — | 215.32 ms |
| Sampled process RSS peak | 1909.94 MiB | 3266.87 MiB |
| Process RSS after requests/drain | 1240.40 MiB | 2783.46 MiB |

These are process-start timings; OS file-cache state was not controlled.
Main score differences were exactly zero, with identical labels and thresholds.
The main concurrent P95 increased by 17.13 ms and passed the approved 50 ms absolute
allowance; this is a measurable slowdown, not a claim of zero performance cost.
All 12 functional cases passed. Burst admission returned one routed result and
three busy results. The 20 ms timeout run performed exactly one real XLM-R forward,
kept the next request busy, preserved main detection and recovered after completion.
All workers finished with zero admitted/active/pending work, normal exit and no
remaining listener; the lowest available memory across runs was 15.42 GiB.

This was Windows, Python 3.10.19, RTX 5070 Ti, existing lab plus the isolated py3langid
0.4.0 cache. `pip check` found no broken installed requirements, but the actual
environment differs from the declared deployment requirements:

| Package | requirements.txt | Tested lab |
|---|---|---|
| torch | 2.7.1 | 2.9.1+cu128 |
| safetensors | 0.5.3 | 0.7.0 |
| fastapi | 0.115.14 | 0.128.0 |
| pydantic | 2.11.7 | 2.12.5 |
| starlette | 0.46.2 | 0.50.0 |
| uvicorn | 0.35.0 | 0.40.0 |
| py3langid | 0.4.0 | isolated cache only; absent from lab installation |

No package versions were changed. This small local run is not a production latency
distribution or validation of the declared requirements on another deployment.
The owner accepted local completion and deferred deployment questions until needed;
they do not block further local development. Backend mode projection and network
orchestration remain later governed items.
