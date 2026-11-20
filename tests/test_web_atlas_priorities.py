"""`portolan.web.atlas_data`'s theme-folder priority ordering (Phase 3,
`01-spec/actions-and-owners-2026-09-19.md`: "The Atlas theme list ...
shows the same ordering and the same three numbers per theme").

`_theme_priority_order` is a pure reordering of a folder's own `notes`
list against whatever `theme_priorities_for_atlas` returns -- these tests
monkeypatch that one function (which is itself the thing
`test_report_priorities.py` already exercises directly against a real
database) so this file can check the ATLAS side of the contract --
"Atlas order matches report order" -- without needing a live report, a
CRM export, or a database connection at all.
"""

from __future__ import annotations

from portolan.report.models import ThemePriority
from portolan.web import atlas_data


def _priority(
    slug: str, name: str, *, accounts: int, arr: float, regions: list[str]
) -> ThemePriority:
    return ThemePriority(
        theme_slug=slug,
        theme_name=name,
        movement="rising",
        account_slugs=[f"{slug}-acct-{i}" for i in range(accounts)],
        account_count=accounts,
        arr=arr,
        arr_unknown_accounts=0,
        regions=regions,
        rolled_up={},
    )


def test_theme_folder_order_matches_the_ranked_priority_list(monkeypatch) -> None:
    """Given the exact ranked list `compute_theme_priorities` would
    produce for the current report window, the theme folder's own
    `notes` list comes back in that same order, each entry carrying the
    "N accounts, $X ARR, regions, movement" subtitle -- and a theme that
    wasn't ranked at all (untouched this window) keeps its original
    place, appended after every ranked one, with no subtitle."""
    ranked = [
        _priority("theme-b", "Theme B", accounts=5, arr=1_000_000.0, regions=["Midwest"]),
        _priority("theme-a", "Theme A", accounts=2, arr=500_000.0, regions=["Northeast"]),
    ]
    monkeypatch.setattr(atlas_data, "theme_priorities_for_atlas", lambda conn: ranked)

    # The folder's own pre-ordering (whatever `updated DESC` produced) --
    # deliberately in a DIFFERENT order than the ranking, and including
    # one theme (`theme-untouched`) the ranking never mentions at all.
    entries = [
        {"slug": "theme-untouched", "title": "Untouched", "updated": "2026-09-01"},
        {"slug": "theme-a", "title": "Theme A", "updated": "2026-08-20"},
        {"slug": "theme-b", "title": "Theme B", "updated": "2026-08-15"},
    ]

    ordered = atlas_data._theme_priority_order(conn=None, entries=entries)

    assert [e["slug"] for e in ordered] == ["theme-b", "theme-a", "theme-untouched"]
    assert ordered[0]["priority"] == "5 accounts, $1,000,000 in annual revenue in Midwest — rising"
    assert ordered[1]["priority"] == "2 accounts, $500,000 in annual revenue in Northeast — rising"
    assert "priority" not in ordered[2]


def test_theme_folder_order_is_unchanged_when_nothing_is_ranked(monkeypatch) -> None:
    """No report has ever been generated (`theme_priorities_for_atlas`
    returns `[]`) -- the folder's own entries pass through exactly as
    given, no `priority` key added to any of them."""
    monkeypatch.setattr(atlas_data, "theme_priorities_for_atlas", lambda conn: [])
    entries = [{"slug": "theme-x", "title": "Theme X", "updated": "2026-09-01"}]

    ordered = atlas_data._theme_priority_order(conn=None, entries=entries)

    assert ordered == entries
    assert "priority" not in ordered[0]
