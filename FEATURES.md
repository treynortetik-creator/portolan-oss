# Portolan features

What a person using Portolan sees, where each thing lives, and how to prove it works. Written 2026-09-22 from the routes in `portolan/web/routes/` and the templates in `portolan/web/templates/`. If a route or template changes, update the section here in the same commit.

## Proving it works: `scripts/verify.py`

One command checks a running Portolan over plain HTTP and writes the proof to a file:

```
uv run python scripts/verify.py all --base-url https://<your-deployment>.up.railway.app
```

- It prints one line, `PASS 24/24 -> var/proof/<UTC time>-all.json` or `FAIL 22/24 -> ... (failed: ...)`, and exits 1 on any failure.
- The JSON file lists every request: URL, status, time taken, size, and the **markers** it counted on the page. A marker is a piece of the page that only shows up when real data rendered, for example "action rows with an owner and a due date". A page that loads but is empty fails its markers.
- Keys: the org key comes from `var/railway/org-key.txt` and the MCP key from `var/railway/mcp-key.txt` (or `--org-key-file` / `--mcp-key-file`, or the `PORTOLAN_ORG_KEY` / `PORTOLAN_MCP_ORG_KEY` environment variables). They are never printed, never written to the result, never put in a URL.
- It only reads. It logs in, opens pages, and asks the MCP door to list its tools. It never clicks Done or Confirm. Only the opt-in `chat` target asks a question, and each question is a paid model call.
- Targets: `health`, `login`, `home`, `actions`, `review`, `reports`, `report-cards`, `action-note`, `how-it-works`, `atlas`, `search`, `note`, `source`, `coverage`, `graph`, `mcp`, `all`. Each section below names its own.
- Opt-in targets, run only by name and never by `all`: `action-done` (a healthy deployment can have none of what it checks) and `chat` (every question is a paid model call; see Chat below).

**Local proof run** (no live site involved), two shells:

```
uv run python -m tests.verify_seed --port 8765 --workdir /tmp/portolan-verify
uv run python scripts/verify.py all --base-url http://127.0.0.1:8765 \
    --org-key-file /tmp/portolan-verify/org-key.txt --mcp-key-file /tmp/portolan-verify/mcp-key.txt
```

The first command rebuilds the local `portolan_test_verify` database, fills it with the synthetic Halvard corpus and one weekly report (about 2 to 3 minutes), writes throwaway keys to the workdir, and serves the app until Ctrl-C. The automated version is `tests/test_verify_script.py`.

General gotchas:
- Five wrong org keys from one address in a minute lock that address out of login for a minute, doubling up to 15 minutes. A bad key file can do this to you.
- A local app must run with `PORTOLAN_COOKIE_SECURE=false`; over plain `http://` the session cookie is otherwise never sent back and every page after login redirects to `/login`. The login check adds a hint to the result when this happens.
- Search on the live site embeds the query on the server's CPU; a slow check (several seconds) is normal. The default timeout per request is 30 seconds (`--timeout`).

## Chat

The home page. A reader types a question about any customer, product or theme and gets an answer written one sentence at a time. Every sentence carries a chip naming the note it came from and, where there is one, a button that opens the original document at the exact spot. A sentence with nothing real behind it is not shown. The left sidebar keeps the reader's past conversations.

Sub-features:
- Example questions built from the three most recently updated accounts ("What has Cobalt Ridge Medical Center said recently?") on an empty chat.
- Answers stream in sentence by sentence.
- Citation chip: the note's title; hover shows its slug and last-updated date. A "note-level" badge marks a sentence tied to a whole note rather than one recorded claim.
- "Open the original at <locator>" button on each cited claim.
- "Answering with <model>" line under each answer, with a count of sentences withheld for an unverified number and of sentences withheld for citing nothing, so a shortened answer says so.
- Copy and Regenerate on each answer; a failed regenerate keeps the old answer.
- Helpful / Not helpful on each answer, with an optional one-line reason. Not helpful saves the question and answer as a candidate eval case (`eval_candidates`) for a person to confirm the gold answer; changing to Helpful removes an unreviewed one.
- Sidebar: + New chat, rename, archive, unarchive, delete.
- When nothing matches, the answer says "nothing recorded" or, if the search index itself looks broken, says that instead.
- Agent mode (`PORTOLAN_CHAT_MODE`, default `agent`; `single` is the one-shot path): before the answer, a short line beside the pulse says what the chat is doing ("Searching the graph…", "Searching for “…”…", "Reading <note title>…", "Following links from <note title>…", "Checking citations…"). The answer, its citations and its footnote look the same as in single mode. On an error or the 45 s limit the answer comes from the single-shot path instead (01-spec/chat-agent-2026-09-25.md).
- Follow-up questions (agent mode): the agent sees the conversation's last three exchanges and the notes they cited, and its first search runs on the previous question plus this one, so "Which of those…" works (`HISTORY_TURNS` in `portolan/web/chat_agent.py`; scenario `tests/cli_scripts/follow_up.json`).

Entry points: `/` (nav "Chat", and the Portolan wordmark), `/c/<conversation id>` for one conversation.

Driving it:
- Preconditions: logged in; the graph has notes.
- Action: open `/`, type a question, press Send.
- `verify.py home` checks the page only: the composer is there, the signed-in name shows in the nav, and at least one example question was built from the graph. It does not ask a question, because an answer is a paid model call.
- `verify.py chat` (opt-in, paid) asks questions the way the browser does and follows every citation. `--question "..."` (repeatable), `--questions-file scripts/demo_questions.txt` (the demo script's ten, one per line), `--expect-nothing "..."` for a question whose right answer is "Nothing recorded". All questions go into one conversation under the `--editor` name (default `verify.py`), so the demo sidebar stays clean. Per question it checks: an answer arrived; it carries a citation (or is "Nothing recorded" for `--expect-nothing`); every citation resolves (the note page returns 200 and holds the cited claim, the original returns 200 and shows the block at the cited spot, and the claim's quote is on that page). Sentences withheld for an unverified number are counted, not failed. It records the answer text, every citation, time to first sentence and to the full answer, and prints one line per question: `q1 PASS 14.2 s, answered, 6 sentences, 9 citations, 0 withheld: <question>`.
- Demo check: `uv run python scripts/verify.py chat --base-url <site> --questions-file scripts/demo_questions.txt --expect-nothing "What is our NPS with Ashgrove Health?"` (ten paid calls).
- Local proof, no model call: start the seed with `--chat-stub` (`uv run python -m tests.verify_seed --port 8765 --workdir /tmp/portolan-verify --chat-stub`); the stub answers each question with claims from the retrieved notes, word for word.
- Expected: the home check passes. To prove an answer on a live site, run `verify.py chat`, or ask one by hand and click a chip through to the note and the original.

Gotchas:
- Chat history belongs to the display name typed at login, not a person. Two people who log in as "Sam" share one history.
- 500 conversations per name at most; delete or archive to start more.
- The model-call meter (`usage_events`) logs a question's length, never its text. The pilot activity log (`activity_events`, below) keeps the text, so the dashboard can list unanswered questions; it is deleted after `PORTOLAN_ACTIVITY_RETENTION_DAYS`.

## Note viewer

Every fact in Portolan lives on a note: an account, a person, a product, a signal, a theme, an action, a report. The note page shows the note's title, type and freshness, a short synthesis, and the list of claims, each with the quote it came from and a button to open the original. It also shows what links to the note and every earlier version.

Sub-features:
- Header: type and status badges, updated date (coloured fresh or stale), created date, stale-after date, version number.
- Account notes show their owner; account and person notes show their open actions.
- Action notes confirmed from a researched card show "Why": the reasoning, the evidence and what was checked.
- An action closed from a "Proposed done" card shows "Done": who confirmed it and when, the later claim that answered it with "Open the original", and what the system checked.
- Synthesis, rendered as formatted text with links to other notes.
- Claims: text, quote, source and locator, fresh or superseded badge, "Open the original" button, Edit.
- Related, Backlinks, Version history.
- Edit the synthesis or a single claim (`/notes/<slug>/edit`); the change is signed with the editor's name and kept in the history.
- Download the note as markdown (`/notes/<slug>.md`). An action confirmed from a researched card downloads what its page shows: owner, account, due and status, Why, Evidence with source links, What the system checked, who confirmed it, and the evidence claims. An action closed from later evidence adds a Done section with the answering claim and its source link.
- All notes index at `/notes`: filter by title or type, 50 per page, and a link to the Company context note.

Entry points: `/notes/<slug>`, `/notes`, every citation chip in chat, every link in Atlas, reports, Actions and Review.

Driving it:
- Preconditions: logged in; at least one signal note with a claim.
- Action: open any note from Atlas or a chat chip.
- `verify.py note` opens the first note in Atlas's Signals folder (or `--note-slug`) and checks it has a title, at least one claim, and at least one "Open the original at" link.
- Expected: all three markers found.

Gotchas:
- Editing writes a new version; there is no undo button, but the old text stays in Version history.

## Source viewer

The last link in the provenance chain. A source page describes one original document (call, email, ticket, feed item, CRM row, PDF): its type, date, extraction status, who took part, which claims cite it and at which spot, and which notes came from it. "Open original" shows the document itself, with the cited block highlighted and scrolled into view.

Sub-features:
- Source page: type, extraction status, source date, ingested date, content fingerprint, participants.
- "Cited from": every claim citing this source, each with a button to its spot in the original.
- "Notes extracted from this source".
- Original viewer: call transcripts by timestamp, emails with headers and paragraphs, tickets by comment, feed items, documents by page and paragraph, CRM rows one field per line (`key=value`).
- `?at=<locator>` highlights and scrolls to the cited block.
- PDFs open in the browser's own viewer at the cited page.

Entry points: `/sources/<slug>`, `/originals/<slug>?at=<locator>`, every "Open the original" button in chat, notes, Atlas, reports and Review.

Driving it:
- Preconditions: logged in; the original file is present in the store.
- Action: from a note, click "Open the original at ...".
- `verify.py source` follows the first such link on the `note` target's page: the source page must have an Open original button and at least one citing claim, and the original must contain a block whose locator matches the one the claim cited.
- Expected: all markers found.

Gotchas:
- Opening an original or a source page writes one access-log row (spec sec 12: every original opened is logged). `verify.py source` does too.
- The highlight is done in the browser; the check proves the cited block exists, not that it is yellow.
- A CRM row is one block at its export locator (`crm/opportunities.csv#O-3001`), highlighted as a whole. Its fields are listed in key order, not the export's column order, so a quote spanning several fields is on the page field by field rather than as one string; `verify.py chat` checks such a quote field by field.

## Atlas

The browsable map of every note, grouped into plain-word folders (Accounts, People, Products, Signals, Themes, Actions, Reports and so on), with a search box on top. Typing filters the tree as you go and shows "Best matches" from the fused search. Opening a note shows its properties, synthesis, claims and links in the right-hand pane, with the tree left open where the note lives.

Sub-features:
- Folder tree with counts; large folders link to "Show all N" in the notes index.
- Signals grouped under their account.
- Themes show their priority line (accounts touched, revenue, regions, movement).
- Search as you type: the tree narrows to matches and "Best matches" appears on top.
- Search results page: result count, cards with snippet and updated date; a "nothing recorded" or "index looks broken" message when empty.
- Note pane: properties, synthesis, claims with "Open the original", edges (links in and out), backlinks, version history, Edit and Download .md.
- An account's pane lists its open actions (a person's pane, the actions they own): what, owner or account, due date with an overdue badge, and the one-line why, each opening the action's own pane. An action confirmed from a researched card shows its Why there.

Entry points: nav "Atlas" (`/atlas`), `/atlas/<slug>`, `/atlas?q=<query>`, and the old `/search?q=<query>`, which now redirects to Atlas.

Driving it:
- Preconditions: logged in; the graph has notes.
- Action: open Atlas, expand a folder, type in the search box.
- `verify.py action-note` takes the first action on `/actions` with a researched why and checks its `.md` download (Why, Evidence lines with a source link, What the system checked, the confirmed-by line, no boilerplate synthesis), its Atlas pane (a Why card), and its account's Atlas pane (at least one open action linking into Atlas). It fails on a graph where no card was ever confirmed.
- `verify.py atlas` checks the tree has at least one folder and one note link. `verify.py search` searches for the first account name in the tree (or `--query`), checks `/search` redirects to Atlas, and checks the results page shows a result count of at least 1, at least one result card, and no "index looks broken" warning.
- Expected: both pass. The search needs no paid call: the query is embedded locally.

Gotchas:
- Search is keyword plus meaning plus recency, so a result can match without containing the exact word.

## Actions

The to-do list the graph produces: every open commitment, grouped by the person who owns it, with the account, the due date, a link to the source, and Done and Dropped buttons. Overdue items are flagged. Below, a list of what was closed this week.

Sub-features:
- Summary line: open count, overdue count, as-of date.
- One group per owner, alphabetical, Unassigned last; within a group, soonest due first.
- Each row: what, account, due date with an overdue badge, source link, one-line "why" when researched.
- Done and Dropped buttons; each click is signed with the reader's name.
- "Closed this week" with who owned it and whether it was done or dropped.
- Empty state: "No open actions right now."

Entry points: nav "Actions" (`/actions`); "Open the Actions page" from an account or person note.

Driving it:
- Preconditions: logged in; at least one open action with an owner and a due date.
- Action: open `/actions`.
- `verify.py actions` checks the summary line counts at least one open action and at least one row sits under a named owner (not Unassigned) with a due date and a link to its note.
- Expected: both markers found. The check never presses Done or Dropped.

Gotchas:
- Overdue is judged against today's date in the deployment's time zone (`PORTOLAN_TIMEZONE`, default America/Phoenix).
- Done and Dropped are immediate; a second click on an already-closed action does nothing.

## Review queue

Where Portolan parks anything it was not confident enough to decide alone: a name it could not place, a name that could be two records, a theme it thinks is resolved, a proposed owner, a proposed action, a report sentence it threw out over an unproven number. Facts are never dropped; they wait here for a person. Each card says in plain words what happened, shows the evidence, says why it matters, and offers the choices.

Sub-features:
- Open count, and a filter strip with a count per kind: Unknown name, Ambiguous name, Proposed resolution, Proposed owner, Proposed action, Proposed done, Possible contradiction, Unsupported quote, Rejected draft, Bad CRM data, Tidy-up failed, Needs a look.
- Evidence on each card: the claim, the quote with surrounding context, the source.
- Choices per kind, for example "This is another name for..." (type-ahead record picker), "Ignore this mention", "Confirm resolution" or "Reject", "Confirm owner", and for a proposed action "Confirm" with owner and due date, or "Dismiss" with a reason.
- Confirming a proposed action creates a real action that then shows on Actions.
- "Proposed done": a later claim on the same account, said by someone on our own side, that a judge read as answering an open action (a reply, a commitment kept, a resolution). The card shows the claim with its source link and names the action. "Confirm done" marks the action done, signed with the reader's name, and keeps the claim on the action; "Dismiss" (with an optional reason) leaves it open and that claim is never proposed for that action again. Nothing closes an action without a person. At most one open card per action. Made by `python -m portolan.report propose-done --week <YYYY-Www>` (`--dry-run` lists the claims it would judge, no judge call and no writes; `--action <id>` checks one action) or `weekly --propose-done`; both off unless run.
- Technical details folded away on each card.
- 25 cards per page, oldest first; the kind filter carries across pages.
- One card per unknown or ambiguous name: a later source naming it joins that card ("also came up in N other sources; one answer covers all of them"), with the sources under Technical details.
- A fact that names only our own side (the company, a branch, one of its people) from a source with no account shows as "Needs a look" with the fact, instead of being lost.

Entry points: nav "Review" (`/review`), `/review?kind=<kind>`.

Driving it:
- Preconditions: logged in; at least one open review row.
- Action: open `/review`, pick a card, choose.
- `verify.py review` checks the eyebrow shows at least one open row and at least one card carries a kind badge.
- `verify.py action-done` (opt-in) opens `/review?kind=Proposed done` and checks for at least one card, one showing the answering claim with a link to its original, and one linking the action it would close.
- Expected: all markers found. The checks never choose anything.

Gotchas:
- An empty queue is a real, healthy state, but `verify.py review` fails on it: the check exists to prove the queue renders rows. Run it where rows exist, or read the failure as "nothing waiting".
- Dismiss, Reject and Ignore ask for confirmation in the browser first.
- A "Proposed done" card whose claim was edited or retracted after it was proposed refuses Confirm and stays open; Dismiss it. An action counts as created on the date of the document it cites, so only claims dated after that day can answer it.

## Weekly reports

One report per week, written into the graph as a note like any other. It opens with an executive summary (what happened this week, what needs attention, by account, suggested actions), then themes by movement (new, rising, falling, resolved), account waves, standing questions, outliers, actions and coverage. Every figure is computed by code; every sentence cites its source.

Sub-features:
- Reports list, most recently updated first, with a Download .md button each.
- Report page: covered week, generated date, the body with links to every note it mentions and "Open the original" links on citations.
- "Suggested actions": proposed action cards from the researcher, shown with their state (proposed with a confidence, confirmed with owner and due date, or dismissed), then free-text suggestions.
- Download as markdown (`/reports/<slug>.md`).
- Each suggested action card's state (proposed, confirmed by whom with owner and due, dismissed) is read from the review queue when the page or the download is opened, so a Confirm on `/review` shows at once; the stored report text is not rewritten.

Entry points: nav "Reports" (`/reports`), `/reports/report-<YYYY>-W<ww>`.

Driving it:
- Preconditions: logged in; the weekly job (`python -m portolan.report weekly ...`) has written at least one report.
- Action: open Reports, click the newest week.
- `verify.py reports` checks the list has at least one weekly report link, opens the newest week (or `--report-slug`), and checks for the "Weekly report" title, at least one body section, at least one action-card line under Suggested actions, and the Download .md link.
- `verify.py report-cards` reads the open action cards on `/review` and checks that no card line on the newest report is out of step with them: "proposed" only while the card is still open, confirmed or dismissed only once it is not.
- Expected: all markers found.

Gotchas:
- "Newest" means the latest week number in the list, not the most recently regenerated.
- A quiet week with no proposed actions fails the action-card marker. Pass `--report-slug` to check a specific week.

## Monday email

What changed this week on one person's top accounts (25 by open pipeline, else ARR), from CRM exports and call transcripts only. Per account: CRM field changes found by code between exports (each also a claim on the account's "CRM changes" note, cited to the newer and the older row), "Does the CRM match the calls?" when the CRM moved a deal forward but a recent call said otherwise, new call moments, and actions that went overdue or were opened. Accounts are ranked by a points score; accounts with no change are one line at the bottom. Every line links to `/originals/<source>?at=<row or timestamp>`. An optional two-sentence summary per account comes from a model and is kept only when it cites the lines above and adds no number. Design: `01-spec/monday-email-2026-09-25.md`.

Entry points: files only. `python -m portolan.report monday --owner <crm id or name> --week <YYYY-Www> --out <dir> [--top N | --accounts A,B] [--base-url URL] [--llm openrouter --max-usd 1]` writes `monday-<owner>-<week>.html`, `.txt` and `.json`. `--send --to <address>` refuses unless `PORTOLAN_SMTP_URL` is set. CRM exports go in through the pipeline (snapshots always; changes too with `PORTOLAN_CRM_CHANGE_CLAIMS=true`) or `python -m portolan.crm ingest <dir>`; `python -m portolan.crm changes` writes changes not yet written.

Driving it:
- Preconditions: dated CRM exports (an `export_date` column or `-YYYY-MM-DD` in the file name) for at least two weeks. The demo graph (`python -m portolan.web.demo`) has them for 2026-08-14 and 2026-08-28 (`corpus/demo/weekly-crm/`).
- Action: `python -m portolan.report monday --owner sam-okafor --week 2026-W35 --out var/proof/monday --base-url <site>`, then open every link in the `.json` `links` list while logged in.
- Expected (demo): 6 of 12 accounts changed, 23 CRM changes, 3 CRM-versus-call checks; every link 200 with its cited block and quote on the page. Automated: `tests/test_report_monday.py`.

Gotchas:
- A CRM file with no date is taken in but not compared: it cannot be placed in the week sequence.
- The email reads changes already written; if its footer counts unwritten changes, run `python -m portolan.crm changes`.

## Public signals (off by default)

Public, second-hand information about each account: posts on its LinkedIn company page (through an RSS.app feed the maintainer sets up; never a person's profile), news feeds, and changes on a few pages of its own website. A weekly scan keeps only concrete events an account team would act on (leadership change, funding, acquisition, layoffs, new product, pricing, expansion, security incident, hiring, partnership) and files each as a cited claim on the account's "Public signals" note. The weekly report adds "Public signals this week", grouped by account. Design and costs: `01-spec/external-signals-2026-09-23.md`.

Sub-features:
- Per-account targets from a YAML file: RSS.app LinkedIn company-page feeds (the company page URL is required), news feeds, and web pages.
- Feed items are read once, by GUID; web pages are snapshotted and diffed section by section, with dates and counters ignored.
- A relevance judge writes one sentence and a verbatim quote per event; code checks the quote and every number against the item.
- Each claim says it is public and who published it, and "Open the original" shows the feed item or the website change record, with both snapshots' times.
- Coverage shows `signals_feed` and `signals_web` rows: seen, kept, filtered as not relevant, refused by robots.txt.

Entry points: `python -m portolan.signals targets import FILE`, `targets list`, `scan`, `status`; the note `/notes/signal-public_signal-<account>` (Atlas: Signals, under the account); the weekly report's "Public signals this week" section.

Driving it:
- Preconditions: `PORTOLAN_EXTERNAL_SIGNALS=true`; accounts in the registry; targets imported; an OpenRouter key for `--llm openrouter`.
- Action: `python -m portolan.signals scan --llm openrouter --budget-usd 1`, then build the weekly report.
- No `verify.py` target of its own. `verify.py note --note-slug signal-public_signal-<account>` checks the note, its claims and an "Open the original" link; `verify.py reports --report-slug report-<week>` checks the report.
- Expected: the scan prints its counts as JSON; the note lists the claims; the report has the section.

Gotchas:
- Off unless the flag is set: `scan` exits 2 with a message, and the report has no section.
- robots.txt is obeyed for every request, so a site that blocks unknown bots (Mayo Clinic's CDN did on 2026-09-23) is skipped, and Google News RSS search is refused.
- The first scan of a page is its baseline; changes show from the second scan on.

## Coverage

The ingest ledger: for each day and connector, how many sources were seen, extracted, partially extracted, failed, sampled and filtered, and why. Below, every source that failed or only partly extracted, and anything waiting to be retried. The page's promise: nothing is silently dropped.

Sub-features:
- Ledger table, last 14 days by default; `?days=N` for up to ten years.
- "Not ingested (failed / partial)" with attempts and last error, linking to each source page.
- "Dead letter (pending redrive)".

Entry points: nav "Coverage" (`/coverage`), `/coverage?days=N`.

Driving it:
- Preconditions: logged in; at least one ingest run recorded.
- Action: open `/coverage`.
- `verify.py coverage` opens `/coverage?days=3650` and checks for at least one ledger row plus the Not ingested and Dead letter sections.
- Expected: all markers found.

Gotchas:
- The check reads all history (`days=3650`) so a quiet fortnight does not fail it; the default page shows 14 days.

## Graph

A picture of the whole graph: every note a dot, every link a line, in 2D or a rotating 3D view. Click a dot to read the note in a side panel; search to find one; filter by note type.

Sub-features:
- 2D and 3D modes, Fit, Rotate.
- Search notes by title.
- Filter by note type; toggle type colours.
- Note panel with the note's body and previous/next.

Entry points: nav "Graph" (`/graph`, `/graph?mode=2d`).

Driving it:
- Preconditions: logged in; the graph has linked notes.
- Action: open `/graph`.
- `verify.py graph` checks the page carries the graph canvas and that `/graph.json` returns at least one node and one edge.
- Expected: both pass. The drawing itself happens in the browser and is not checked.

Gotchas:
- Shows the 2,000 most recently updated notes at most.

## How it works

A public, self-contained explainer page with narration audio: what Portolan is and how a customer's words become a cited fact. No login needed, so it can be sent to anyone.

Sub-features:
- Narrated walkthrough.
- A link back into the app.

Entry points: nav "How it works" (logged in or not), the link on the login page, `/how-it-works`.

Driving it:
- Preconditions: none.
- Action: open `/how-it-works` without logging in.
- `verify.py how-it-works` fetches it with no session and checks for the audio, the link back to the app, and a full-size file (over 100 KB).
- Expected: all markers found.

Gotchas:
- About 2.8 MB. Browsers cache it for an hour.
- It is the file `01-spec/how-portolan-works.html`; a build that leaves out `01-spec/` serves a 404 here.

## MCP door

The organisation's one door in for its own AI tools. Claude Code or Claude Desktop connects to `/mcp` with a shared key and can then search the graph, open notes and sources, follow links, check coverage, look up precedents and register standing questions for the weekly report. It is the same graph and the same citations the web app shows.

Sub-features (tools): `search`, `open_note`, `open_source`, `graph_neighbors`, `coverage`, `precedents`, `add_standing_question`, `list_standing_questions`.

Entry points: `https://<host>/mcp`. Connection steps for Claude Code, Claude Desktop, and (with OAuth configured) claude.ai and ChatGPT connectors are in `docs/mcp-hosted.md`. With OAuth, `/.well-known/oauth-protected-resource/mcp` and `/.well-known/oauth-protected-resource` serve the public metadata that names the identity provider.

Driving it:
- Preconditions: `PORTOLAN_MCP_ORG_KEY` set on the service (32 characters or more); the key in `var/railway/mcp-key.txt` or `--mcp-key-file`.
- Action: connect a client with `Authorization: Bearer <key>` and list tools.
- `verify.py mcp` checks that a request with no key is refused (401), then with the key: `initialize` returns a session and the server name `portolan`, `tools/list` returns the tools including `search`, `open_note` and `open_source`, and the session closes cleanly. The full tool list is recorded in the result file.
- Expected: all checks pass. No tool is called.

Gotchas:
- Without `PORTOLAN_MCP_ORG_KEY` (or one shorter than 32 characters) and without OAuth, `/mcp` does not exist (404).
- OAuth is off unless `PORTOLAN_MCP_OAUTH_ISSUER` is set; an incomplete OAuth setup stays off and logs `MCP OAuth is configured but incomplete`. With OAuth on, the no-key 401 carries a `WWW-Authenticate` header, and each refusal logs `mcp auth failed: ip=... reason=...`.
- If the transport fails to start, `/mcp` answers 503 and the rest of the app keeps running; the deploy log says `MCP transport failed to start`.
- The web login does not open the MCP door, and the MCP key does not open the web pages.
- Each `verify.py mcp` run puts one `mcp auth failed` warning in the server log: that is the no-key probe, on purpose.
- `docs/mcp-hosted.md` lists seven tools; the server has eight (`precedents` was added since).

## Pilot usage

For the 30-day pilot : who uses the app, and whether answers help. Under the shared key any signed-in reader can open it; under work-account sign-in only admins can ("Login and health").

Sub-features:
- Activity log (`activity_events`, migration 0037): page views (route template and slug, never the query string), chat questions (text, answer length, citations, withheld sentences, latency, cost, outcome), clicks from an answer's "Open the original" through to the original, review decisions, action confirms, email opens and clicks. No IP address or user agent.
- Dashboard `/admin/usage`: last 30 days and 8 weeks of active users, questions per user, thumbs-up share, citation click-through, email opens and clicks, per-person counts, and the top questions whose answer had no citation.
- Email tracking for the Monday email: `portolan.activity.email_links.email_open_url` (a 1x1 GIF at `/t/o/<token>.gif`) and `email_click_url` (`/t/c/<token>`, redirects only to the signed target). Tokens are signed with the session secret and count for 90 days.
- Retention: rows older than `PORTOLAN_ACTIVITY_RETENTION_DAYS` (default 180) are deleted at web startup and by `python -m portolan.activity purge`. `python -m portolan.activity delete-user "<name>"` removes one person's activity, feedback and eval candidates.

Entry points: `/admin/usage` (no nav link), the Helpful / Not helpful buttons under each chat answer.

Driving it:
- Local proof, no model call: start the seed with `--chat-stub`, ask a question, open an original from the answer, press Not helpful, open `/admin/usage`. Run on 2026-09-25: `var/proof/usage-2026-09-25/`.

Gotchas:
- Under the shared key names are typed at login, so the per-person table counts names, not verified people; under work-account sign-in each row is a verified `Name <email>` (`01-spec/pilot-sign-in-2026-09-25.md`).
- A click is counted only from the "Open the original" button; the note chip is not tracked.

## Login and health

Not features a reader goes looking for, but every other check stands on them.

- Login (`/login`): organisation key plus a display name (2 to 40 characters) that signs edits and names the chat history. Sessions last 8 hours. Log out is in the nav. `verify.py login` checks that a page without a session redirects to `/login`, the form has both fields, and posting the key sets the session cookie and redirects home.
- Work-account sign-in (off by default; `PORTOLAN_WEB_AUTH=workos`): `/login` goes to the WorkOS AuthKit sign-in page and back through `/auth/callback`; only members of the pilot's WorkOS organization (and, optionally, verified emails at allowed domains) get in. The verified `Name <email>` signs edits, review decisions, action confirms, chat history and usage events. Every page needs sign-in except `/healthz`, static files, `/signed-out`, the email pixel and click link, and `/mcp` (its own auth). Log out also ends the WorkOS session. People are editors by default; an admin (`PORTOLAN_ADMIN_EMAILS`) can make someone a viewer (reads and asks, cannot change the graph) on `/admin/people`, and only admins open `/admin/usage`. Setup steps: `01-spec/pilot-sign-in-2026-09-25.md`. Proof without a WorkOS account: `uv run python scripts/web_signin_proof.py` (fake AuthKit on loopback; run on 2026-09-25: `var/proof/web-signin-2026-09-25/`). `verify.py login` posts the org key, so it cannot sign in while this is on.
- Health (`/healthz`): public, runs one database query, answers `{"status": "ok"}` or a 503. Railway's health check points here. `verify.py health` checks the body.
