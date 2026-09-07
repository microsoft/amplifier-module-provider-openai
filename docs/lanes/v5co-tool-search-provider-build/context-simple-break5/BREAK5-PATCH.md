# Break 5 — `context-simple` compaction must not drop loaded-tool state

**Status: DONE as a verified patch artifact, NOT as a commit in that repo — because
this lane cannot reach that repo. That is a GOAL DEFECT, reported below, not a gap
this lane absorbed.**

---

## 1. The goal defect, stated plainly

`GOAL.md` requires, as a deliverable:

> **`context-simple` break 5 fixed** (G13 compaction survival).

and simultaneously constrains:

> Do not merge anything to main; **do not edit files outside the paths this lane
> owns**. … **Never touch other repos.**

This lane's worktree contains **exactly one repo**:

```
/home/bkrabach/dev/hw-model-performance/lanes/v5co-tool-search-provider-build/
└── amplifier-module-provider-openai        <- the only checkout
```

`amplifier-module-context-simple` is not in it and has no lane branch. So the only
way to satisfy that deliverable as literally written is to write into a repo the
goal forbids touching. `GOAL.md` anticipates precisely this and names the remedy:

> **AND: every option this goal offers you must have at least one target inside the
> paths it says you own.** If the only way to satisfy a deliverable is to write a
> file outside your worktree (another repo, or the manager's unversioned `goals/`
> directory), that is a **DEFECT IN THIS GOAL**, not a task. Report it against the
> goal, ship the patch as an artifact under your ARTIFACT ROOT, and resolve — do not
> edit another repo, and do not invent a fourth outcome branch.

That is what this directory is. **The remedy for the manager is one line: give the
next lane a `context-simple` worktree, or apply this patch directly** (§4).

Note this is a *packaging* defect, not a *scoping* one: the work item is right that
break 5 belongs with breaks 1/3/6 — they are one feature. Only the lane's checkout
was sized for one repo.

---

## 2. What break 5 actually is

`DESIGN.md` w3 §4, break 5 (severity **high**):

> Compaction removes input items. Remove a `tool_search_output` / `additional_tools`
> item and (a) those tools cease to exist (`TS:854`) and (b) the cache breaks
> forward. … `context-simple`'s ladder has no concept of an un-droppable,
> position-pinned item.

`TS:854`, normative:

> "Tools that were not listed as part of this array will not be available to the
> model … **changing the loaded tool set will break the model's cache from that
> point forward.**"

Two failures from one action, and **both are silent**. The model discovers (a) by
calling a tool that no longer exists; nobody discovers (b) at all, except as cost —
and OpenAI's cache is **grow-only** (`00-what-we-know.md` §2a: strict truncation of
a cached prefix returns `0`, MISS), so "breaks forward" means a full cold rebuild,
not a dip.

**Where the state lives in this implementation.** The provider carries the hosted
items on `Message.metadata["openai:tool_search_items"]` (the existing provider-state
channel, same as encrypted reasoning) rather than as free-standing input items. So
"drop the item" is, concretely, "drop the message that carries the metadata" — which
is exactly what `_remove_messages_with_protection` does at ladder levels 3/5/7.

---

## 3. The fix

`break5-loaded-tool-state-protection.patch` (67 lines) against
`amplifier-module-context-simple` @ `a2a098b`.

1. A named, provider-neutral protected set:
   ```python
   LOADED_TOOL_STATE_METADATA_KEYS: frozenset[str] = frozenset({"openai:tool_search_items"})
   ```
   Keyed on **metadata, not on a provider name** — a second provider adding the same
   kind of state adds a key and needs no other change.
2. `_remove_messages_with_protection` adds every carrying message's index to
   `protected_indices`. Because the tool-pair cascade
   (`_try_remove_tool_pair_from_result` / `_from_assistant`) already consults that
   same set, one insertion protects the message from the cascade too.

**Deliberately NOT changed:** `_truncate_tool_result` and `_stub_user_message`
already rebuild with `{**msg, ...}`, so they preserve `metadata` — a truncated
message keeps its loaded-tool record. Only **removal** had to learn this. Verified,
not assumed (§4 shows the whole suite still green).

**Scope honesty — what this does NOT do.** `TS:893`'s *positional* contract for an
`additional_tools` input item is untouched, because this provider never emits that
item into history: it is rebuilt at the input tail on every request from
session-scoped provider state. If a future implementation persists it into the
message list, break 5 acquires a second half (ordering, not just retention) that
this patch does not cover. Said here rather than discovered later.

---

## 4. Verification — run it yourself, ~2 minutes, $0

```bash
git clone https://github.com/microsoft/amplifier-module-context-simple /tmp/cs && cd /tmp/cs
git checkout a2a098b
cp <this-dir>/test_loaded_tool_state_protection.py tests/

# FAIL-BEFORE
uv run pytest -q tests/test_loaded_tool_state_protection.py     # 1 failed, 2 passed

patch -p1 < <this-dir>/break5-loaded-tool-state-protection.patch

# PASS-AFTER
uv run pytest -q tests/test_loaded_tool_state_protection.py     # 3 passed
uv run pytest -q -m "not live"                                  # 61 passed
```

Measured in this lane on 2026-09-06, against `a2a098b`:

| step | result |
|---|---|
| patch applies | clean, `patch -p1`, no fuzz |
| FAIL-BEFORE (unpatched + test) | **1 failed, 2 passed** — `len([]) == 0`: the carrier was removed |
| PASS-AFTER (patched + test) | **3 passed** |
| full suite, unpatched | 58 passed |
| full suite, patched | **61 passed** (58 + 3), 0 regressions |

**The test is guarded against being vacuous**, because the first version of it
*was*: `add_message` is a coroutine, and an un-awaited call left the manager with
zero messages while two of three assertions still "passed". It now asserts
`len(mgr.messages) == 26` after the fixture, asserts that compaction actually
reduced the list, and asserts that ordinary messages are still removable — so the
protection cannot pass by being a blanket do-not-compact.

---

## 5. Files here

| file | what it is |
|---|---|
| `break5-loaded-tool-state-protection.patch` | the fix, `patch -p1` against `a2a098b` |
| `test_loaded_tool_state_protection.py` | the FAIL-BEFORE/PASS-AFTER guard; belongs at `tests/` |
| `BREAK5-PATCH.md` | this file |
