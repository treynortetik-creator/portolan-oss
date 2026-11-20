"""Surfacing: the weekly report built from four mechanical feeds
(`01-spec/surfacing-and-weekly-report.md`, `01-spec/portolan-spec-v2.md` sec 7).

Public surface:

- `portolan.report.diff`: `build_diff`, `quiet_accounts`, `notes_created`,
  `notes_updated`, `superseded_facts`.
- `portolan.report.themes`: `refresh_themes`, `theme_summary`,
  `theme_trend`, `theme_weekly_counts`, `reject_theme_resolution` and
  `confirm_theme_resolution` (a human's dismiss or confirm of a pending
  `theme_resolution` review-queue row).
- `portolan.report.questions`: `answer_standing_questions`.
- `portolan.report.outliers`: `list_outliers`.
- `portolan.report.synthesis`: `refresh_account_synthesis`.
- `portolan.report.weekly`: `build_weekly`, `render_email`.
- `portolan.report.accounts`: CRM-ARR lookup and "which account does this
  claim belong to" helpers (ARR is not stored in the graph).
- `portolan.report.numbers`: `NumberFacts`/`check_numerals`, so a model
  never states a number it wasn't given.

`python -m portolan.report weekly --as-of YYYY-MM-DD [...]` runs the CLI
(`portolan/report/weekly.py` `main()`).
"""

from __future__ import annotations
