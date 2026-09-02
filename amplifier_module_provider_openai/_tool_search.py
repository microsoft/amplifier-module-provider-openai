"""Namespace-form deferred tool loading (OpenAI Responses API `tool_search`).

WHAT THIS IS
------------
OpenAI's Responses API can defer the *parameter schemas and descriptions* of
tools the model has not asked for yet.  Two things activate it, both inside the
top-level ``tools`` array (guide ``TS:9-12``):

1. a ``{"type": "tool_search"}`` entry, and
2. ``defer_loading: true`` on the functions you want deferred.

Discovered tools are injected at the **end** of the context window
(``TS:5``, ``TS:862``; ``prompt-caching.md:283-284``), so discovery is an
*append*, not an edit of the pinned head.  That is the entire reason this lever
is worth building: the tool block is ~47% of our pinned head, and every other
way of shrinking it rewrites the front of the cache.

WHY NAMESPACES ONLY (there is deliberately no "flat" mode)
----------------------------------------------------------
``TS:20``: for an individual deferred *function* the model still sees its name
and description, so flat deferral only defers the parameter schema.  Probe
``bub`` measured both forms head-to-head on ``gpt-5.6-terra``:

    flat  ``defer_loading`` on 14 functions : 8,677 -> 7,407 tok  (saves 1,270)
    namespace form, all members deferred    : 8,677 -> 1,104 tok  (saves 7,573)

Namespaces are **6.0x** better for identical engineering cost and risk, so the
flat form is not offered as a config value at all.

WHAT PROBE ``bub`` MEASURED THAT THIS TABLE ENCODES
---------------------------------------------------
* ``web``, ``browser``, ``python``, ``computer`` are **reserved** namespace
  names -- undocumented, and an HTTP 400 on every request that uses one
  (measured n=18 candidate names).  Hence ``internet`` rather than ``web``.
* The hosted search loads a **whole namespace**, not the single tool asked for
  (measured 2/2 arms: asking for ``glob`` loaded all five ``files`` tools).
  So grouping granularity *is* the cost model, which is why ``delegate`` --
  3,602 tok on its own -- is split out of ``orchestration`` into its own
  ``delegation`` namespace rather than riding in on every ``load_skill`` use.
* ``bash`` (90.7% of sessions) and ``todo`` (83.3%) are used so near-universally
  that deferring them buys a search round-trip in ~9 of 10 sessions to save
  905 tokens.  ``TS:60`` permits mixing deferred and non-deferred members inside
  one namespace, so they ship non-deferred.

OPENAI-ONLY BY CONSTRUCTION
---------------------------
Everything here runs *below* the ``ChatRequest`` seam, inside
``_convert_tools_from_request``.  Nothing above the seam changes, so the
Anthropic path cannot observe this feature existing.  That matters concretely:
changing the tools array destroys 100% of the Anthropic cache (w1-o2), and a
structurally-identical regression already escaped a 151-test unit suite once.
``tests/test_tool_search_namespaces.py`` asserts the containment rather than
trusting it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Config values for the `tool_loading` knob.
TOOL_LOADING_STATIC = "static"
TOOL_LOADING_DEFERRED_NAMESPACE = "deferred_namespace"
TOOL_LOADING_MODES = frozenset({TOOL_LOADING_STATIC, TOOL_LOADING_DEFERRED_NAMESPACE})

# Namespace names OpenAI reserves for its own hosted tool surfaces. Using one
# is an HTTP 400 on EVERY request:
#   "Function 'web.web_fetch' is not allowed in reserved namespace 'web'."
# The vendored 897-line guide never uses the word "reserved" -- this list is
# measured (probe `bub`, n=18 candidate names, <= $0.034), not documented.
RESERVED_NAMESPACE_NAMES: frozenset[str] = frozenset(
    {"web", "browser", "python", "computer"}
)

# Default grouping table. Hand-authored and byte-stable on purpose: a
# *heuristic* subset (recency, task classifier, per-agent) would vary the tools
# block between requests, and probe `12v` measured that a 2.1% byte change
# zeroes the cache outright. Sorted emission (namespace name, then member name)
# means dict iteration order can never leak into the wire bytes either.
DEFAULT_NAMESPACES: tuple[dict[str, Any], ...] = (
    {
        "name": "delegation",
        "description": (
            "Delegate a task to a specialised sub-agent that works "
            "autonomously and reports back."
        ),
        "members": ("delegate",),
    },
    {
        "name": "files",
        "description": ("Read, write, edit, search and list files in the workspace."),
        # `apply_patch` is listed here for the FUNCTION-shaped case only. When
        # the native engine is active the provider emits a bare
        # `{"type": "apply_patch"}`, which never reaches the grouping step at
        # all -- native/hosted shapes are passthrough by construction. Listing
        # it keeps the function-shaped fallback out of the "unlisted tool"
        # warning path on every request.
        "members": (
            "apply_patch",
            "edit_file",
            "glob",
            "grep",
            "read_file",
            "write_file",
        ),
    },
    {
        "name": "internet",
        "description": "Search the public web and fetch the contents of a URL.",
        "members": ("web_fetch", "web_search"),
    },
    {
        "name": "knowledge",
        "description": (
            "Load skill knowledge packages, run recipes, and switch runtime modes."
        ),
        "members": ("load_skill", "mode", "recipes"),
    },
    {
        "name": "shell",
        "description": "Run shell commands and manage a task checklist.",
        "members": ("bash", "todo"),
    },
)

# Members that are never deferred. See module docstring: deferring these costs a
# search round-trip in ~9 of 10 sessions to save 905 tokens.
DEFAULT_ALWAYS_LOADED: tuple[str, ...] = ("bash", "todo")

# Hosted tool-search items the API emits in `response.output`. Both arrive with
# `execution: "server"` and `call_id: null` (measured). They MUST survive the
# round trip: per `TS:854`, dropping a `tool_search_output` item makes those
# tools cease to exist for the model *and* breaks the cache forward.
HOSTED_TOOL_SEARCH_ITEM_TYPES: frozenset[str] = frozenset(
    {"tool_search_call", "tool_search_output"}
)

# Unlisted-tool sets already warned about (see build_namespaced_tools).
_WARNED_UNLISTED: set[tuple[str, ...]] = set()


class ToolSearchConfigError(ValueError):
    """Raised at mount time for an unusable `tool_loading` configuration.

    Fails loud at construction rather than per-request: a bad namespace table
    is a deployment mistake, and discovering it as an HTTP 400 on every request
    of a live session is strictly worse than refusing to start.
    """


def validate_tool_loading(mode: Any) -> str:
    """Normalise and validate the `tool_loading` config value."""
    if mode is None or mode == "":
        return TOOL_LOADING_STATIC
    if not isinstance(mode, str) or mode not in TOOL_LOADING_MODES:
        raise ToolSearchConfigError(
            f"tool_loading must be one of {sorted(TOOL_LOADING_MODES)}; "
            f"got {mode!r}. Note there is deliberately no 'flat' mode: "
            "flat defer_loading was measured at 1,270 tokens saved vs 7,573 "
            "for the namespace form (probe bub, gpt-5.6-terra)."
        )
    return mode


def normalize_namespaces(raw: Any) -> tuple[dict[str, Any], ...]:
    """Validate a namespace table and freeze it into deterministic order.

    Returns namespaces sorted by name, each with its members sorted by name, so
    two processes with the same config always emit byte-identical tool blocks.
    """
    if raw is None:
        return DEFAULT_NAMESPACES
    if not isinstance(raw, list) or not raw:
        raise ToolSearchConfigError(
            "tool_namespaces must be a non-empty list of "
            "{name, description, members} mappings."
        )

    seen_names: set[str] = set()
    seen_members: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []

    for entry in raw:
        if not isinstance(entry, dict):
            raise ToolSearchConfigError(
                f"tool_namespaces entries must be mappings; got {type(entry).__name__}."
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ToolSearchConfigError("each tool_namespaces entry needs a 'name'.")
        if name in RESERVED_NAMESPACE_NAMES:
            raise ToolSearchConfigError(
                f"namespace name {name!r} is RESERVED by OpenAI for a hosted tool "
                "surface and returns HTTP 400 on every request. Reserved names: "
                f"{sorted(RESERVED_NAMESPACE_NAMES)}. (Use 'internet' for web tools.)"
            )
        if name in seen_names:
            raise ToolSearchConfigError(f"duplicate namespace name {name!r}.")
        seen_names.add(name)

        members = entry.get("members") or ()
        if isinstance(members, str) or not hasattr(members, "__iter__"):
            raise ToolSearchConfigError(
                f"namespace {name!r} 'members' must be a list of tool names."
            )
        members = tuple(str(m) for m in members)
        if not members:
            raise ToolSearchConfigError(f"namespace {name!r} has no members.")
        # TS:22 -- "aim to keep each namespace to fewer than 10 functions".
        if len(members) >= 10:
            logger.warning(
                "[PROVIDER] tool namespace %r has %d members; OpenAI recommends "
                "fewer than 10 per namespace (TS:22). Search quality may degrade.",
                name,
                len(members),
            )
        for member in members:
            if member in seen_members:
                raise ToolSearchConfigError(
                    f"tool {member!r} appears in two namespaces "
                    f"({seen_members[member]!r} and {name!r}); a tool belongs to "
                    "exactly one namespace."
                )
            seen_members[member] = name

        normalized.append(
            {
                "name": name,
                "description": str(entry.get("description") or ""),
                "members": tuple(sorted(members)),
            }
        )

    return tuple(sorted(normalized, key=lambda ns: ns["name"]))


def build_namespaced_tools(
    flat_tools: list[dict[str, Any]],
    namespaces: tuple[dict[str, Any], ...],
    always_loaded: frozenset[str],
) -> list[dict[str, Any]]:
    """Regroup an already-converted flat OpenAI tool list into the namespace form.

    Input is exactly what `_convert_tools_from_request` produces today, so the
    static path and the deferred path share one conversion and cannot drift.

    Emission order is fixed and total -- native passthrough tools in their
    original relative order, then namespaces sorted by name (members sorted by
    name), then any unlisted function tools sorted by name, then the
    `tool_search` entry last. Ordering is a cache-affecting setting (`PC:42`),
    so it must never depend on dict iteration order.

    A tool that is not in the table falls through as a plain, undeferred
    function tool and is logged at WARNING. Failing open is deliberate: a tool
    that silently vanished into a namespace would be exactly the class of
    silent downgrade this provider already warns about elsewhere.
    """
    by_name: dict[str, dict[str, Any]] = {}
    native_passthrough: list[dict[str, Any]] = []

    for tool in flat_tools:
        if not isinstance(tool, dict):  # pragma: no cover - defensive
            native_passthrough.append(tool)
            continue
        # Native/hosted shapes (`{"type": "computer"}`, `{"type": "apply_patch"}`,
        # web_search_preview, ...) are vendor-owned. TS says nothing about
        # nesting them in a namespace, so they are never namespaced and never
        # deferred -- they pass through byte-identical.
        if tool.get("type") != "function" or "name" not in tool:
            native_passthrough.append(tool)
            continue
        by_name[tool["name"]] = tool

    emitted: set[str] = set()
    namespace_items: list[dict[str, Any]] = []

    for ns in namespaces:
        members: list[dict[str, Any]] = []
        for member_name in ns["members"]:
            tool = by_name.get(member_name)
            if tool is None:
                # A table entry with no registered tool behind it is normal
                # (roster varies by bundle) -- not an error, and silently
                # skipping keeps the emitted block a function of what is
                # actually registered.
                continue
            member = {
                "type": "function",
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            }
            # TS:18 -- defer_loading applies to the functions inside a
            # namespace, not to the namespace object itself.
            if member_name not in always_loaded:
                member["defer_loading"] = True
            members.append(member)
            emitted.add(member_name)
        if not members:
            continue
        namespace_items.append(
            {
                "type": "namespace",
                "name": ns["name"],
                "description": ns["description"],
                "tools": members,
            }
        )

    unlisted = sorted(name for name in by_name if name not in emitted)
    if unlisted and tuple(unlisted) not in _WARNED_UNLISTED:
        # Once per distinct unlisted set per process: the condition is a
        # deployment mistake to fix once, not a per-request event. Warning
        # every request would bury it in its own noise.
        _WARNED_UNLISTED.add(tuple(unlisted))
        logger.warning(
            "[PROVIDER] tool_loading=deferred_namespace: %d tool(s) are not in the "
            "namespace table and are being sent flat and undeferred: %s. Add them to "
            "`tool_namespaces` to include them in the deferred block.",
            len(unlisted),
            ", ".join(unlisted),
        )

    result: list[dict[str, Any]] = list(native_passthrough)
    result.extend(namespace_items)
    result.extend(by_name[name] for name in unlisted)
    # TS:9-12 -- the tool_search entry is what activates the whole mechanism.
    result.append({"type": "tool_search"})
    return result


def build_additional_tools_item(
    tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Build the developer-role `additional_tools` INPUT item (`TS:870-890`).

    This is the append-shaped escape hatch for a tool that appears *mid-session*
    (a mode contributing tools, a skill activating one). It lands in the `input`
    array, so it never touches the top-level `tools` block and never rewrites
    the front of the cache -- the whole reason V-80 flagged it.

    `TS:893`: tools in an `additional_tools` item become available only after
    that item appears in the input, so once emitted its position must never
    move. Callers append at the current tail and leave it there.
    """
    if not tools:
        return None
    return {
        "type": "additional_tools",
        "role": "developer",
        "tools": [dict(t) for t in tools],
    }


def _item_type(item: Any) -> str | None:
    if isinstance(item, dict):
        return item.get("type")
    return getattr(item, "type", None)


def _to_plain_dict(item: Any) -> dict[str, Any] | None:
    """Best-effort conversion of an SDK output item to a replayable dict.

    `openai==2.8.1` has no model for a `tool_search_call` / `tool_search_output`
    output item, so it parses both as a generic `ResponseOutputMessage` with the
    real payload surviving in pydantic extras (measured, probe `bub` G8). The
    SDK does not *fail* -- it mis-types silently -- so this reader must go
    through `model_dump()`/`__dict__` rather than named attributes.
    """
    if isinstance(item, dict):
        return dict(item)
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        try:
            data = dump(exclude_none=False)
        except Exception:  # noqa: BLE001 - pragma: no cover, defensive around SDK variance
            data = None
        if isinstance(data, dict):
            return data
    data = getattr(item, "__dict__", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return None


def extract_hosted_tool_search_items(output_items: list[Any]) -> list[dict[str, Any]]:
    """Collect `tool_search_call` / `tool_search_output` items, in wire order.

    Without this the items are dropped by the two
    ``if block_type not in {"tool_call", "function_call"}: continue`` guards in
    the response path, and then per ``TS:854`` every discovered tool ceases to
    exist on the next request and the model re-searches every single turn.
    """
    captured: list[dict[str, Any]] = []
    for item in output_items:
        if _item_type(item) not in HOSTED_TOOL_SEARCH_ITEM_TYPES:
            continue
        as_dict = _to_plain_dict(item)
        if as_dict is None:
            logger.warning(
                "[PROVIDER] hosted tool-search item could not be serialised for "
                "replay; discovered tools will be re-searched next turn (TS:854)."
            )
            continue
        # `status` is response-scoped bookkeeping; everything else round-trips
        # verbatim, including `tools[]` (the loaded set) and the null `call_id`.
        captured.append(
            {k: v for k, v in as_dict.items() if v is not None or k == "call_id"}
        )
    return captured
