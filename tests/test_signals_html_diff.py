"""Page text extraction and the noise-tolerant section diff (`portolan.signals.html_text`,
`portolan.signals.diff`). Pure functions, no database, no network."""

from __future__ import annotations

from portolan.signals.diff import content_hash, diff_snapshots, normalize_paragraph
from portolan.signals.html_text import Section, extract_page, html_to_text

PAGE = """
<html><head><title>Leadership | Example Health</title><script>var x = 1;</script></head>
<body>
  <header><nav><a href="/">Home</a><a href="/careers">Careers</a></nav></header>
  <main>
    <h1>Our leadership</h1>
    <p>Example Health is led by a team of physicians and administrators.</p>
    <div class="cookie" hidden>We use cookies to improve your experience.</div>
    <h2>Executive team</h2>
    <ul>
      <li>Jordan Reyes, President and CEO</li>
      <li>Sam Patel, Chief Financial Officer</li>
    </ul>
    <p>Last updated September 2, 2026. 1,204 followers on social media.</p>
    <aside>Related: Patient stories</aside>
    <h2>Board of directors</h2>
    <p>The board meets quarterly.</p>
  </main>
  <footer>&copy; 2026 Example Health</footer>
</body></html>
"""


def test_extract_page_scopes_to_main_and_splits_by_heading() -> None:
    page = extract_page(PAGE)

    assert page.title == "Leadership | Example Health"
    headings = [s.heading for s in page.sections]
    assert headings == ["Our leadership", "Executive team", "Board of directors"]
    executive = page.sections[1].paragraphs
    assert "Jordan Reyes, President and CEO" in executive
    assert "Sam Patel, Chief Financial Officer" in executive
    text = " ".join(p for s in page.sections for p in s.paragraphs)
    for chrome in ("Home", "Careers", "cookies", "Patient stories", "var x"):
        assert chrome not in text
    assert page.needs_browser is False


def test_an_inline_svg_title_is_not_the_page_title() -> None:
    html = (
        "<html><head><title>Newsroom | Example</title></head><body><main>"
        '<svg><title>Facebook</title><path d="M0"/></svg><h2>News</h2><p>Story.</p>'
        "</main></body></html>"
    )
    assert extract_page(html).title == "Newsroom | Example"


def test_extract_page_without_main_drops_header_and_footer() -> None:
    html = (
        "<body><header>Site banner text</header><h2>News</h2><p>Opened a new clinic.</p>"
        "<footer>Footer links</footer></body>"
    )
    page = extract_page(html)

    assert [s.heading for s in page.sections] == ["News"]
    assert page.sections[0].paragraphs == ["Opened a new clinic."]


def test_extract_page_flags_a_javascript_app_shell() -> None:
    html = '<html><body><div id="root"></div><script src="/app.js"></script></body></html>'
    assert extract_page(html).needs_browser is True


def test_html_to_text_flattens_a_feed_description() -> None:
    fragment = '<p>Example Health <b>named</b> a new CFO.</p><p><a href="x">Read more</a></p>'
    assert html_to_text(fragment) == "Example Health named a new CFO.\nRead more"


def test_normalize_paragraph_ignores_dates_times_counters_and_copyright() -> None:
    a = "Last updated September 2, 2026 at 10:15 AM. 1,204 followers. © 2026 Example"
    b = "Last updated September 23, 2026 at 4:40 PM. 1,391 followers. © 2027 Example"
    assert normalize_paragraph(a) == normalize_paragraph(b)
    # Plain numbers and prices are not noise.
    assert normalize_paragraph("Visits cost $49") != normalize_paragraph("Visits cost $59")


def _sections(page_html: str) -> list[Section]:
    return extract_page(page_html).sections


def test_noise_only_change_is_no_change_and_hashes_equal() -> None:
    later = PAGE.replace("September 2, 2026", "September 23, 2026").replace("1,204", "1,390")
    old, new = _sections(PAGE), _sections(later)

    assert content_hash(old) == content_hash(new)
    assert diff_snapshots(old, new, url="https://example.org/leadership", title="t") is None


def test_added_leader_is_reported_in_its_section() -> None:
    later = PAGE.replace(
        "<li>Sam Patel, Chief Financial Officer</li>",
        "<li>Sam Patel, Chief Financial Officer</li><li>Alex Kim, Chief Nursing Officer</li>",
    )
    change = diff_snapshots(
        _sections(PAGE), _sections(later), url="https://example.org/leadership", title="t"
    )

    assert change is not None
    assert [(c.heading, c.added, c.removed) for c in change.changes] == [
        ("Executive team", ["Alex Kim, Chief Nursing Officer"], [])
    ]
    text = change.render(observed="2026-09-23 14:00 UTC", previous="2026-09-16 14:00 UTC")
    assert "Section: Executive team" in text
    assert "Added: Alex Kim, Chief Nursing Officer" in text


def test_replaced_leader_shows_both_sides() -> None:
    later = PAGE.replace("Sam Patel, Chief Financial Officer", "Dana Wu, Chief Financial Officer")
    change = diff_snapshots(_sections(PAGE), _sections(later), url="u", title="t")

    assert change is not None
    only = change.changes[0]
    assert only.added == ["Dana Wu, Chief Financial Officer"]
    assert only.removed == ["Sam Patel, Chief Financial Officer"]


def test_a_paragraph_moved_between_sections_is_not_a_change() -> None:
    old = [
        Section("News", ["Opened the Mesa clinic.", "Named a new CMO."]),
        Section("Archive", ["Old story."]),
    ]
    new = [
        Section("News", ["Named a new CMO."]),
        Section("Archive", ["Opened the Mesa clinic.", "Old story."]),
    ]
    assert diff_snapshots(old, new, url="u", title="t") is None


def test_a_removed_section_is_reported() -> None:
    old = [Section("Pricing", ["Standard visit: $120"]), Section("About", ["We care."])]
    new = [Section("About", ["We care."])]
    change = diff_snapshots(old, new, url="u", title="t")

    assert change is not None
    assert [(c.heading, c.removed) for c in change.changes] == [
        ("Pricing", ["Standard visit: $120"])
    ]
