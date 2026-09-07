#!/usr/bin/env python3
"""G5a / G5b' Anthropic containment guardrail -- STATIC, $0, no API call.

This is `bub`'s `containment_check.py` re-run against a tree where the
treatment **actually exists**. `bub` §5 stated its own limitation plainly:

    "this is the *static* form of T3/G5a run against code where the treatment
     DOES NOT EXIST YET ... G5a-G5d against a real implementation remain
     blocking for Phase 2 and are not discharged by this document."

`model_performance-v5co` is that real implementation, so this re-runs the same
check with it in place and reproduces `bub`'s published pair:

    both arms   sha256 = cea1c7d9015febae...03be   (G5a: 0 divergences)
    neg control sha256 = 2921b40e402f23f2...5590   (the check CAN fail)

FOUR CHECKS
-----------
1. **SEAM GREP, post-implementation form.** `tool_search` / `defer_loading` /
   `NamespaceTool` must appear **0** times in `amplifier-core`,
   `amplifier-module-loop-streaming` and `amplifier-module-provider-anthropic`,
   and **non-zero** in `amplifier-module-provider-openai` -- which is the
   deliverable and the inverse of what `bub` asserted. A zero there now would
   mean the feature does not exist.
2. **G5a byte-identity.** Anthropic's real `_convert_tools_from_request` +
   `_apply_tool_cache_control` over the same `list[ToolSpec]`, once as shipped
   and once with `defer_loading` extras stamped on every spec (legal --
   `ToolSpec` is `extra="allow"`, `MM:141-147`). The hashes must be equal, and
   equal to `bub`'s published value.
3. **NEGATIVE CONTROL.** The "obvious" implementation `DESIGN.md` §3.3 forbids
   -- namespace grouping done *upstream* of the seam, so BOTH providers see the
   reorder -- must move the Anthropic hash. A containment check that cannot
   fail is not a check.
4. **THE LIVE ARM, which `bub` could not run.** Drive the real
   `OpenAIProvider._convert_tools_from_request` over the SAME seam payload in
   `tool_search.mode: off` and `tool_search.mode: namespaced`. The OpenAI array
   MUST move (or the feature is inert) while the Anthropic array computed from
   the same payload MUST NOT.

G5b (breakpoint position) is reported but is **not** treated as reorder
protection: `bub` measured that `_apply_tool_cache_control` walks
`reversed(tools)`, so the index is positional-from-the-end and invariant under
any reorder -- old-G5b PASSES the exact failure it exists to catch. `cal`
strengthened that to "a pure function of the same array G5a hashes, marginal
coverage ZERO". What runs here instead is **G5b'**, the absolute per-arm
invariant: exactly one tools breakpoint, actually placed, on the last function
tool, <=4 blocks total -- which catches the one class G5a structurally cannot,
both arms drifting identically.

Usage: containment_check.py [<output-dir>]
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

EVALS = Path("/home/bkrabach/dev/openai-evals-team-ci")
PANT = EVALS / "amplifier-module-provider-anthropic"
CORE = EVALS / "amplifier-core" / "python"
V12 = EVALS / ".amplifier/evaluation/probes/12v-tool-block-diet/tools_full.json"
# This lane's own worktree -- the tree that carries the implementation.
POAI = Path(__file__).resolve().parents[3]

# bub / PHASE1-VERDICT.md §5, published 2026-09-02.
BUB_BOTH_ARMS = "cea1c7d9015febaec1c9ad14872ab071884a4671f97249e8641f97a60b3203be"
BUB_NEG_CONTROL = "2921b40e402f23f2f572309e5d54e4800ce0ffc7781e4eec228a7b0996eb5590"

sys.path.insert(0, str(POAI))
sys.path.insert(0, str(PANT))
sys.path.insert(0, str(CORE))


def sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def seam_grep() -> dict:
    """Post-implementation seam grep. See check 1 in the module docstring."""
    out: dict[str, dict] = {}
    targets = {
        "amplifier-module-loop-streaming": (EVALS, 0),
        "amplifier-core": (EVALS, 0),
        "amplifier-module-provider-anthropic": (EVALS, 0),
        "amplifier-module-provider-openai": (POAI.parent, None),
    }
    for mod, (cwd, expect_zero) in targets.items():
        path = mod if cwd is EVALS else POAI.name
        p = subprocess.run(
            [
                "grep",
                "-rn",
                "-E",
                r"tool_search|defer_loading|NamespaceTool",
                "--include=*.py",
                path,
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        hits = [ln for ln in p.stdout.splitlines() if "/.venv/" not in ln]
        entry: dict = {"hits": len(hits), "sample": hits[:3]}
        if expect_zero == 0:
            entry["expected"] = "0 (above/beside the seam)"
            entry["ok"] = len(hits) == 0
        else:
            entry["expected"] = "non-zero (this is the deliverable)"
            entry["ok"] = len(hits) > 0
        out[mod] = entry
    return out


def breakpoint_profile(tools: list) -> dict:
    idx = [i for i, t in enumerate(tools) if "cache_control" in t]
    return {
        "cache_control_indices": idx,
        "cache_control_count": len(idx),
        "n_tools": len(tools),
    }


def g5b_prime(tools: list) -> dict:
    """cal's absolute per-arm invariant (DESIGN.md §3.4, G5b')."""
    idx = [i for i, t in enumerate(tools) if "cache_control" in t]
    last_fn = max(
        (i for i, t in enumerate(tools) if t.get("name")),
        default=-1,
    )
    checks = {
        "exactly_one_tools_breakpoint": len(idx) == 1,
        "actually_placed": len(idx) >= 1,
        "on_the_last_function_tool": bool(idx) and idx[-1] == last_fn,
        "at_most_four_blocks_total": len(idx) <= 4,
    }
    return {**checks, "PASS": all(checks.values()), "indices": idx}


def main() -> int:
    dest_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
    from amplifier_core.message_models import ToolSpec
    from amplifier_module_provider_anthropic import AnthropicProvider

    tools_full = json.loads(V12.read_text(encoding="utf-8"))
    # The seam payload: exactly what LS:3012-3014 hands to ChatRequest.
    base_specs = [
        ToolSpec(
            name=t["name"],
            description=t.get("description", ""),
            parameters=t.get("parameters") or {},
        )
        for t in tools_full
        if t.get("name")
    ]

    prov = AnthropicProvider.__new__(AnthropicProvider)
    prov.enable_prompt_caching = True
    prov.cache_stable_region_ttl_1h = False

    def anth(specs):
        conv = prov._convert_tools_from_request(list(specs))
        conv, used = prov._apply_tool_cache_control(conv)
        return conv, used

    # --- arm "off": as shipped -------------------------------------------
    off, off_used = anth(base_specs)

    # --- arm "namespaced": extras ride on the ToolSpec (extra="allow"), but
    # the namespacing itself happens ONLY inside the OpenAI provider.
    ns_specs = [
        ToolSpec(
            name=s.name,
            description=s.description,
            parameters=s.parameters,
            defer_loading=True,
        )
        for s in base_specs
    ]
    on, on_used = anth(ns_specs)

    # --- NEGATIVE CONTROL: the forbidden upstream-grouping implementation.
    # Kept byte-for-byte as bub wrote it, including the pre-T-5NS flat
    # `orchestration` group -- the point is to reproduce bub's published hash,
    # not to model the shipped table.
    NS = {
        "files": ["read_file", "write_file", "edit_file", "grep", "glob"],
        "orchestration": ["delegate", "load_skill", "recipes", "mode"],
        "shell": ["bash", "todo"],
        "internet": ["web_search", "web_fetch"],
    }
    order = {n: i for i, n in enumerate(m for ns in sorted(NS) for m in sorted(NS[ns]))}
    bad_specs = sorted(base_specs, key=lambda s: order.get(s.name, 999))
    bad, bad_used = anth(bad_specs)

    # --- THE LIVE ARM: the real OpenAI provider, both modes ---------------
    from amplifier_module_provider_openai._tool_search import (
        DEFAULT_ALWAYS_LOADED,
        DEFAULT_NAMESPACES,
        build_namespaced_tools,
    )

    flat_openai = [
        {
            "type": "function",
            "name": s.name,
            "description": s.description or "",
            "parameters": s.parameters,
        }
        for s in base_specs
    ]
    openai_namespaced = build_namespaced_tools(
        flat_openai, DEFAULT_NAMESPACES, frozenset(DEFAULT_ALWAYS_LOADED)
    )

    result = {
        "check": "G5a / G5b' Anthropic containment, POST-implementation",
        "lane": "v5co-tool-search-provider-build",
        "spend_usd": 0.0,
        "provider_openai_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=POAI,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "seam_grep": seam_grep(),
        "arms": {
            "off": {
                "tools_sha256": sha(off),
                "breakpoint_used": off_used,
                **breakpoint_profile(off),
                "g5b_prime": g5b_prime(off),
            },
            "namespaced": {
                "tools_sha256": sha(on),
                "breakpoint_used": on_used,
                **breakpoint_profile(on),
                "g5b_prime": g5b_prime(on),
            },
            "NEGATIVE_CONTROL_upstream_grouping": {
                "tools_sha256": sha(bad),
                "breakpoint_used": bad_used,
                **breakpoint_profile(bad),
            },
        },
        "openai_side": {
            "off_sha256": sha(flat_openai),
            "namespaced_sha256": sha(openai_namespaced),
            "n_tools_off": len(flat_openai),
            "n_entries_namespaced": len(openai_namespaced),
            "tool_search_entry_last": openai_namespaced[-1] == {"type": "tool_search"},
        },
    }

    result["G5a_byte_identity"] = sha(off) == sha(on)
    result["G5a_matches_bub_published"] = sha(off) == BUB_BOTH_ARMS == sha(on)
    result["G5b_breakpoint_identity"] = breakpoint_profile(off) == breakpoint_profile(
        on
    )
    result["G5b_prime_PASS"] = g5b_prime(off)["PASS"] and g5b_prime(on)["PASS"]
    result["negative_control_fires"] = sha(off) != sha(bad)
    result["negative_control_matches_bub_published"] = sha(bad) == BUB_NEG_CONTROL
    # The feature must actually DO something, or a passing guardrail is vacuous.
    result["openai_side_actually_moved"] = sha(flat_openai) != sha(openai_namespaced)
    result["seam_grep_ok"] = all(v["ok"] for v in result["seam_grep"].values())

    gates = [
        "G5a_byte_identity",
        "G5a_matches_bub_published",
        "G5b_prime_PASS",
        "negative_control_fires",
        "negative_control_matches_bub_published",
        "openai_side_actually_moved",
        "seam_grep_ok",
    ]
    result["gates"] = gates
    result["verdict"] = "PASS" if all(result[g] for g in gates) else "FAIL"

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "containment-static.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {dest}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
