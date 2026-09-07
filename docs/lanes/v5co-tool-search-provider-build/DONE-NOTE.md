# DONE-NOTE — lane `v5co-tool-search-provider-build`

**Work item:** `model_performance-v5co` (project `model_performance`)
**Repo:** `microsoft/amplifier-module-provider-openai`, branch `lane/v5co-tool-search-provider-build`, from `main` @ `f104e6c`
**Outcome branch:** **A — RESOLVED.** Every deliverable is DONE. Nothing was recorded NOT-POSSIBLE; the cap never bound.
**Spend:** **$0.010574 of the $10 authority** (0.11%). Residue $9.989426.

---

## 1. Headline — read this first

**The build already existed and had been reverted for a reason that was about evidence, not code.**

`cal` and the manager both grepped `tool_search` in `provider-openai`, got **0 hits**, and correctly concluded the feature was absent. It is absent. What neither could see from a grep is **why**: lane `6ei` built this exact machinery, it shipped as **PR #82 (`39aae65`)**, and **PR #83 (`7a8da74`)** reverted it three hours later under *"main carries wins only"* — because `6ei`'s own pre-registered quality gate `E5_quality` was **NOT MEASURED** ("pre-registered as out of cap") and tripwire **D2 (tool substitution) FIRED 2/23** in the deferred arm vs 0/13 control.

That verdict was correct **and it is exactly what `cal`'s Phase 2 exists to buy.** So this lane did not rebuild from the design; it **restored `6ei`'s measured implementation onto current `main`**, renamed to the config surface the work item and `DESIGN.md` §2.2(a) both name (`tool_search.mode`, not `6ei`'s `tool_loading`), dropped the R0-interaction test (R0 was also reverted, correctly, for a measured 2.8–3.3× cost regression), and then **found and fixed a break nobody knew about** (§3).

**Cost consequence:** the $47 Phase 2 unblock cost **$0.0106**, not because the work was small, but because it was already paid for once.

**Priority note (`GOAL.md`: "if two objectives read as equally required, treat the FIRST as the objective"):** the first objective is `tool_search.mode` implemented in `provider-openai`. Everything else was reachable too, so no objective was dropped.

---

## 2. Deliverables

| # | Deliverable | State | Evidence |
|---|---|---|---|
| 1 | `tool_search.mode` implemented; `grep -rn tool_search --include='*.py'` non-zero | **DONE** | **0 → 80** hits in the package (0 → 156 across all tracked `*.py`) (§6) |
| 2 | Breaks **1, 3, 6** fixed; break 3 covered by a **FAIL-BEFORE** test | **DONE** | **6 failed / 0 passed at `f104e6c`** → **0 failed / 6 passed** on branch (§4) |
| 3 | `context-simple` **break 5** fixed | **DONE as a verified patch artifact** — *goal defect, reported* (§5) | FAIL-BEFORE 1 failed/2 passed → PASS-AFTER 3 passed; suite 58 → 61 |
| 4 | Break **4 NOT touched**, PR says why | **DONE** | §7; PR body; `DESIGN.md` §1.8 measured false |
| 5 | Smoke run: hosted pair survives a turn, accepted back as input | **DONE** | `smoke-tool-search.json`, **10/10 checks PASS**, live `gpt-5.6-terra` (§3) |
| 6 | **Anthropic non-regression guardrail** — reproduce `cea1c7d9…03be` pair + `2921b40e…5590` control | **DONE — exact match** | `containment-static.json`, verdict **PASS**, **$0** (§8) |
| 7 | Re-derived measurements justified | **DONE — nothing re-derived** | §9 |
| 8 | Suite green, ruff clean, **draft** PRs, no merge | **DONE** | 825 → **867 passed**; ruff 9 → **8** findings (§6) |
| 9 | DONE-NOTE at lane artifact root, never repo root | **DONE** | this file |

---

## 3. The finding — break 6 has a second half, and it wedges the feature at turn 2

**This is the most important thing in this note.**

The smoke run's turn 1 succeeded. Turn 2 came back **HTTP 400**:

```
Missing namespace for function_call 'glob'. It does not exist in the default
namespace. Round-trip the model's function_call item with its namespace field
included.                                          [param: input[3].namespace]
```

`DESIGN.md` §2.2(f)1 read `TS:391-397` as a **dispatch** concern only:

> "our tool names are globally unique, so `tools.get(tool_call.name)` keeps working — **provided the provider passes `name` through unchanged and does not concatenate `namespace/name`**"

That is **necessary and not sufficient.** The API *also* requires the `namespace` echoed back on the replayed `function_call` **input** item. `cal`'s break-6 note ("`function_call` carries `name:"glob"` **AND** `namespace:"files"` with name unqualified, so `tools.get(name)` keeps working") is correct as far as it goes and stops one step short of this.

**Without the fix the feature works for exactly one turn and then wedges** — and `6ei` shipped, and #82 merged, without this being caught, because unit tests agreed with their own mocks. **This is what a live smoke run is for**, and it is the single strongest argument in this lane for keeping deliverable 5 funded in future goals.

**Fix.** `ToolCall` has no namespace field, so the value rides the provider-state metadata channel as `openai:tool_call_namespaces` (`{call_id: namespace}`), captured on both response paths and stamped back at the **single** `function_call` emission site — which covers all 7 branches that build one. Fallback when metadata is gone (compaction dropped the carrying message — break 5 — or the transcript predates the fix): derive from the configured table, which is *the same table that produced the tools block*, so it is exact rather than a guess. `off` mode never stamps a field it never had; the default byte-identity test pins that.

Pinned by 5 unit tests. Confirmed on the wire: turn 2 replayed `{"name": "glob", "namespace": "files"}` and returned 200.

### Smoke run, full result (`smoke-tool-search.json`, verdict **PASS**, 10/10)

| | observed |
|---|---|
| tools block | **6 entries** = 5 namespaces `[delegation, files, internet, knowledge, shell]` + `{"type":"tool_search"}` **last** — the **T-5NS split table**, `internet` not `web` |
| break 1 | `tool_choice` requested `"required"` → **`"auto"` on the wire**, with the warning |
| break 3 capture | `[tool_search_call, tool_search_output]` on `metadata` |
| break 6 hosted | `call_id: null` ×2, `execution: "server"` ×2 |
| break 6 dispatch | `function_call.name == "glob"` — unqualified |
| break 3 re-emission | pair replayed **verbatim** (byte-identical), **HTTP 200** |
| break 6 round trip | `function_call` replayed **with `namespace: "files"`** |
| turn 2 output | `"OK"` |
| usage | t1 851 in / 50 out · t2 887 in / 5 out |

**One incidental observation, recorded not banked (n=1):** `tool_search_output.tools[]` came back naming the **namespace** (`"files"`), not the individual members. Consistent with `cal`'s measured "`{paths:[ns]}` returns the whole namespace (6/6)" — the model emitted the paths-only shape here. Not a new measurement; n=1, one prompt shape.

---

## 4. FAIL-BEFORE / PASS-AFTER, quoted both ways

`tests/test_break3_tool_search_round_trip.py` deliberately **imports nothing this change adds** — it drives the public provider path and refers to the metadata key by its literal string. A collection error would only prove a module is absent; these are real assertion failures.

```
at main f104e6c (source stashed, test present):   6 failed,  0 passed
on lane/v5co-tool-search-provider-build:          0 failed,  6 passed
```

One earlier draft of that file **passed vacuously** (`[] == []` when nothing is captured). It now asserts `len(captured) == 2` and `len(replayed) == 2` *before* comparing. Recorded because the same trap caught the break-5 test too (§5).

---

## 5. Break 5 — DONE as a patch artifact, and the goal defect that forced it

`GOAL.md` requires break 5 fixed in `context-simple`, *and* forbids touching other repos. **This lane's worktree contains only `amplifier-module-provider-openai`.** `GOAL.md` names this case explicitly and prescribes: report it against the goal, ship the patch as an artifact under the artifact root, resolve. Done.

→ `docs/lanes/v5co-tool-search-provider-build/context-simple-break5/` (patch + test + `BREAK5-PATCH.md`)

**Measured against `amplifier-module-context-simple` @ `a2a098b`:**

| step | result |
|---|---|
| `patch -p1` | applies clean, no fuzz |
| FAIL-BEFORE | **1 failed, 2 passed** (`len([]) == 0` — the carrier was removed) |
| PASS-AFTER | **3 passed** |
| full suite unpatched → patched | **58 → 61 passed**, 0 regressions |

**Remedy for the manager, one line:** give the next lane a `context-simple` worktree, or apply the patch directly. It is a *packaging* defect, not a scoping one — the work item is right that break 5 belongs with 1/3/6.

**Honesty note.** The first version of that test was **vacuous**: `add_message` is a coroutine, and un-awaited calls left the manager with **zero messages** while 2 of 3 assertions still "passed". It now asserts the fixture landed (`len(mgr.messages) == 26`), that compaction actually ran, and that ordinary messages are *still* removable — so the protection cannot pass by being a blanket do-not-compact. Found by inspection during verification, not by the suite.

**Scope stated, not implied:** `TS:893`'s *positional* contract for `additional_tools` is untouched, because this provider rebuilds that item at the input tail each request rather than persisting it into history. A future implementation that persists it acquires a second half of break 5 this patch does not cover.

---

## 6. Suite, ruff, grep

| | main `f104e6c` | this branch |
|---|---|---|
| `pytest -q -m "not live"` | **825 passed**, 2 deselected | **867 passed**, 2 deselected (+42) |
| `ruff check amplifier_module_provider_openai/` | **9 findings** | **8 findings** |
| `git grep -c tool_search -- 'amplifier_module_provider_openai/*.py'` | **0** | **80** |
| `git grep -c tool_search -- '*.py'` (all tracked) | **0** | **156** |

**Ruff:** zero new findings. The rule/file set is identical to `main`'s with line numbers shifted, minus one — the pre-existing `PERF402` at the `function_call` emission loop is *resolved* by the break-6 fix (the loop now has a body). All 8 remaining findings pre-date this branch. There is no ruff config in `pyproject.toml` and CI runs `pytest` only, so "ruff clean" is read as "adds no finding"; both new source files and all lane scripts pass a clean `ruff check` and `ruff format --check` on their own. No collateral reformatting of other files (`git diff --stat` confirms).

---

## 7. Break 4 — decided, not overlooked

**Not implemented, deliberately.** `openai==2.8.1` **serializes AND parses** the hosted items without exception; it silently coerces both to `ResponseOutputMessage` with payloads intact (`bub` G8, re-confirmed by `cal`). That is precisely why `_to_plain_dict` reads through `model_dump()` / `__dict__` rather than named attributes. `DESIGN.md` §1.8's claim that *"response parsing is the hard blocker: 2.8.1 has no response model"* is **measured false** — the section already carries the correction inline. A 2.8.1 → 3.x bump touches every module importing `openai` and invalidates the SDK-2.4x fixtures this provider's own tests pin, for **zero** benefit. This lane's live smoke run on `openai==2.8.1` is a fourth independent confirmation: both hosted items round-tripped through the shipped SDK.

**Break 2 (`parallel_tool_calls`)** is likewise untouched — `cal` closed it: both values return 200 and search normally; the vendor's 8/8 examples are a doc convention, not a requirement.

---

## 8. Anthropic guardrail — PASS, exact reproduction, $0

The one thing `GOAL.md` says must not be dropped.

| arm | sha256 | vs `bub` published |
|---|---|---|
| `off` | `cea1c7d9015febaec1c9ad14872ab071884a4671f97249e8641f97a60b3203be` | **exact match** |
| `namespaced` | `cea1c7d9015febaec1c9ad14872ab071884a4671f97249e8641f97a60b3203be` | **exact match** |
| negative control (forbidden upstream grouping) | `2921b40e402f23f2f572309e5d54e4800ce0ffc7781e4eec228a7b0996eb5590` | **exact match** |

**G5a 0 divergences. Negative control fires — the check can fail, so its pass means something.**

Also run, and this is the part `bub` could not:

- **`bub` §5 stated its own limitation**: *"this is the static form of T3/G5a run against code where the treatment DOES NOT EXIST YET … G5a–G5d against a real implementation remain blocking for Phase 2."* This lane **is** that real implementation, so the same check now runs with it in place.
- **Seam grep, post-implementation form** (the inverse of `bub`'s): **0 hits** in `amplifier-core`, `loop-streaming`, `provider-anthropic`; **non-zero** in `provider-openai`. A zero there now would mean the feature does not exist.
- **The live arm `bub` could not run**: the OpenAI array **must** move (13 flat tools → 6 entries) while the Anthropic array from the same seam payload **must not**. Both hold. Without this, a guardrail that passes because the feature is inert would look identical to one that passes because containment works.
- **G5b′, not old G5b.** `bub` measured that `_apply_tool_cache_control` walks `reversed(tools)`, so the breakpoint index is positional-from-the-end and **invariant under any reorder** — old-G5b *passes the exact failure it exists to catch* (index stayed `[12]`, count `1`, under the forbidden reorder). `cal` strengthened that to "a pure function of the same array G5a hashes ⇒ marginal coverage **zero**." What runs here is `cal`'s **G5b′** absolute per-arm invariant — exactly one tools breakpoint, actually placed, on the last function tool, ≤4 total — **PASS both arms**. It catches the one class G5a structurally cannot: both arms drifting identically.

**The hash did not move. The build ships.**

### Cap arithmetic — the goal over-priced this by ~100×

`GOAL.md` priced the guardrail at *"~$1.00"* inside a ~$1.20 expectation. **The guardrail is a static, deterministic, offline hash computation and costs $0.00** — `DESIGN.md` §3.4 says so in as many words ("G5a–G5c are cheap and deterministic and should run in CI on every commit"), and `bub` ran it for $0. Reporting the deviation because `GOAL.md`'s authoring rule requires arithmetic to be checkable on first read:

```
stated:    ~4 wire requests x ~$0.05  +  guardrail pair ~$1.00     = ~$1.20 expected, $10 authority
observed:   5 billed requests = $0.010574  +  guardrail $0.00      =  $0.010574 actual
```

The over-pricing was **harmless here** (it sized the authority generously and nothing was skipped) and is recorded only so a future goal does not price a deterministic CI check as if it were a run. Residue **$9.989426** — ample; nothing was left unbought.

**Observed price before the bulk spend**, as required: first billed request was 851 in / 50 out ≈ **$0.0023**, at which point the remaining plan (≤4 more requests) was obviously fundable.

---

## 9. Nothing was re-derived — `cal`'s $0.52 was reused, not re-bought

| measurement | source | how it was reused |
|---|---|---|
| Reserved namespace names (`web`/`browser`/`python`/`computer` → HTTP 400) | `bub`, n=18 names | Encoded as `RESERVED_NAMESPACE_NAMES`, **refused at construction**; `internet` ships. Never probed. |
| T-5NS split table (`delegation` separate from `knowledge`; +18 tok; NET-WORSE 14.8% → 0.0%) | `cal` | Shipped verbatim as `DEFAULT_NAMESPACES`; asserted by `test_delegate_is_split_into_its_own_namespace`. |
| Loading granularity is model-driven; E[loaded] is a **band [2,002 , 3,006]** | `cal`, n=13, 0 counter-examples | Quoted in the README. **Not re-measured** — the smoke's n=1 observation is recorded as incidental, explicitly not banked. |
| Head-saving table 8,677 → 1,925 tok (36.6%) | `bub` / `12v` | Quoted in the README as measured, sourced. |
| `openai==2.8.1` round-trips hosted items | `bub` G8 + `cal` | Break 4 not touched (§7). |
| Anthropic G5a/negative-control hashes | `bub` §5 | **Reproduced** (that was the deliverable), not re-derived — same script shape, same published constants asserted. |

**Zero measurements were re-bought.** The only new measurement is the break-6 namespace 400 (§3), which no prior probe could have produced because no prior probe reached turn 2 through the provider.

---

## 10. Spend ledger

| item | requests | tokens | cost |
|---|---|---|---|
| smoke attempt 1 — turn 1 | 1 | 851 in / 50 out | $0.002302 |
| smoke attempt 1 — turn 2 | (rejected, HTTP 400 — **the finding**) | — | $0 |
| smoke attempt 2 — turns 1+2 | 2 | 1,738 in / 55 out | $0.004136 |
| smoke attempt 3 (final, recorded) — turns 1+2 | 2 | 1,738 in / 55 out | $0.004136 |
| Anthropic guardrail | 0 (static) | — | **$0.00** |
| unit suite, ruff, break-5 verification | 0 | — | $0.00 |
| **TOTAL** | **5 billed** | **4,327 in / 160 out** | **$0.010574** |

Rates: `gpt-5.6-terra` $2.00/M in, $12.00/M out (`_cost.py`, verified 2026-09-01).
**Authority $10 · spent $0.010574 · residue $9.989426.** No DTU created, no infrastructure registered, nothing to tear down.

---

## 11. What remains open (for the manager, and for Phase 2)

1. **`cal`'s Phase 2 can now launch.** G9, G10, G11, G4′, G12, G13, G5a-live, G5c, G5d were structurally unbuyable; they are now measurable. Use `tool_search.mode: namespaced` (the config key changed from `6ei`'s `tool_loading: deferred_namespace`).
2. **The quality hazard that caused the #83 revert is still unpriced.** D2 (tool substitution — the model reaching for always-loaded `bash` instead of searching for `read_file`) fired **2/23** in `6ei`'s deferred arm vs 0/13 control, and `E5_quality` was never measured. **Restoring this code does not retire that verdict** — it makes the measurement possible. Do not read this lane as evidence the feature should be default-on. It is default-**off** and the README says why, in the same place it states the savings.
3. **Break 5 needs a home** — apply the patch, or give the next lane a `context-simple` worktree (§5).
4. **Break 6's second half is a design correction** worth folding back into `DESIGN.md` §2.2(f)1 and `00-what-we-know.md`: *the namespace must survive the round trip, not just the dispatch.* Not done here — this lane owns one repo and must not edit the evals tree.
5. **`additional_tools` positional contract (`TS:893`)** is exercised by unit tests only. No live request in this lane emitted one, because that needs a mid-session roster change. Untested on the wire; said plainly rather than implied.
