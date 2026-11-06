"""Direct OpenAI lifecycle facts, separate from capabilities and saved pins.

Source (checked 2026-11-06): https://developers.openai.com/api/docs/deprecations.md
Never apply this catalog policy to Azure deployment names or custom gateways.
"""

import re
from datetime import UTC, date, datetime

# Exact aliases plus documented dated snapshots (not arbitrary family prefixes).
_SHUTDOWNS: dict[str, str] = {}
for shutdown, models in (
    ("2026-07-23", (
        "gpt-5-chat-latest", "gpt-5-codex", "gpt-5.1-chat-latest",
        "gpt-5.1-codex", "gpt-5.1-codex-max", "gpt-5.1-codex-mini",
        "gpt-5.2-codex", "o3-deep-research", "o3-deep-research-2025-06-26",
        "o4-mini-deep-research", "o4-mini-deep-research-2025-06-26",
    )),
    ("2026-08-10", ("gpt-5.2-chat-latest", "gpt-5.3-chat-latest")),
    ("2026-10-01", ("gpt-5.4-cyber",)),
    ("2026-10-23", (
        "o1", "o1-2024-12-17", "o1-pro", "o1-pro-2025-03-19",
        "o3-mini", "o3-mini-2025-01-31", "o4-mini", "o4-mini-2025-04-16",
        "gpt-4.1-nano", "gpt-4.1-nano-2025-04-14", "gpt-image-1",
    )),
    ("2026-12-01", (
        "gpt-image-1-mini", "gpt-image-1.5", "gpt-image-1.5-2025-12-16",
        "chatgpt-image-latest",
    )),
    ("2026-12-11", (
        "gpt-5", "gpt-5-2025-08-07", "gpt-5-mini", "gpt-5-mini-2025-08-07",
        "gpt-5-nano", "gpt-5-nano-2025-08-07", "gpt-5-pro", "gpt-5-pro-2025-10-06",
        "o3", "o3-2025-04-16", "o3-pro", "o3-pro-2025-06-10",
    )),
    ("2027-01-14", (
        "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20",
        "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
        "gpt-4.1", "gpt-4.1-2025-04-14", "gpt-4.1-mini", "gpt-4.1-mini-2025-04-14",
    )),
    ("2027-02-01", ("gpt-5.5-cyber-preview",)),
    ("2027-04-01", (
        "gpt-5.1", "gpt-5.1-2025-11-13", "gpt-5.3-codex",
        "gpt-5.4-nano", "gpt-5.4-nano-2026-03-17",
    )),
):
    _SHUTDOWNS.update(dict.fromkeys(models, shutdown))


def shutdown_date(model_id: str, reported: object = None) -> str | None:
    """Prefer a valid catalog date, otherwise return a documented exact date.

    A missing catalog entry is never evidence of shutdown. Unknown IDs remain
    unknown; there is no automatic substitution for explicit model pins.
    """
    if isinstance(reported, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", reported):
        try:
            date.fromisoformat(reported)
        except ValueError:
            pass
        else:
            return reported
    return _SHUTDOWNS.get(model_id)


def is_shutdown(model_id: str, *, reported: object = None, today: date | None = None) -> bool:
    shutdown = shutdown_date(model_id, reported)
    return shutdown is not None and date.fromisoformat(shutdown) <= (
        today or datetime.now(UTC).date()
    )