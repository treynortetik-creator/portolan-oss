# Extraction schema v0

First stab, 2026-09-12, orchestrator session. the maintainer's framing: this will be refined as we test, and customised per customer at onboarding. So v0 has a **base schema** every customer shares and a **customer extension** produced during onboarding. Every output carries `schema_version` so a rebuild knows what extracted what.

## What the extractor receives

The connector hands the model one envelope per source (or per segment of a long source):

| Field | What |
|---|---|
| `source_id` | slug of the source note this came from |
| `source_type` | `call` · `email` · `ticket` · `doc` · `crm` · `chat` · `survey` · `rss` |
| `source_date` | absolute date of the interaction, so "next quarter" can be resolved |
| `participants` | names, roles, and registry ids where the connector already matched them (email domains, call invitees, ticket requester) |
| `registry_context` | the accounts, contacts, products and competitors the connector thinks are relevant, with aliases. Short. |
| `vocabulary` | the customer's own product names, jargon and known competitors (from the customer extension) |
| `text` | the source with **locators injected**: timestamps on transcript lines, message ids on emails, page and paragraph markers on documents, line numbers on anything else |
| `segment` | `{index, total}` when a long source is split |

## What the extractor returns

Strict JSON, enforced by the API's structured output. Nothing else.

```json
{
  "schema_version": "0.1",
  "source_id": "ashgrove-qbr-2026-08-14",
  "summary": "One to three sentences in durable language. Feeds the source note.",
  "entities": [
    {"mention": "Ashgrove Medical", "kind": "account", "registry_id": "ashgrove-medical", "confidence": 0.95},
    {"mention": "Dr. Alvarez", "kind": "contact", "registry_id": null, "confidence": 0.6}
  ],
  "facts": [
    {
      "id": "f1",
      "type": "complaint",
      "claim": "Ashgrove Medical Center received three late deliveries of the X-series pump in August 2026.",
      "entities": ["ashgrove-medical", "x-series-pump"],
      "locator": "00:14:32",
      "quote": "we had three late deliveries last month and nobody called us",
      "speaker": "contact:maria-chen",
      "valid_from": "2026-08-01",
      "valid_to": null,
      "severity": "high",
      "sentiment": -2,
      "confidence": 0.9
    }
  ],
  "actions": [
    {"what": "Send Ashgrove a delivery root-cause summary", "owner_mention": "our rep, Sam", "due": "2026-08-28", "locator": "00:31:10"}
  ],
  "unknown_entities": ["Corvane Medical Center (possibly a competitor)"],
  "coverage": {"segments_read": 1, "segments_total": 1}
}
```

## Fact rules (the extractor is told these, the merge step enforces them)

1. **One claim per fact.** No "and." Split compound statements.
2. **Durable language.** Absolute dates resolved from `source_date`. No "yesterday," "next quarter," "recently."
3. **No pronouns in claims.** Name the entity every time.
4. **Locator is required.** A fact without one is rejected at merge and counted in the coverage report.
5. **Quote is required** for `complaint`, `praise`, `competitor_mention`, `commitment`, `pricing`. Verbatim, forty words or fewer. These are the types where the exact words carry the evidence.
6. **Entities resolve to registry ids** when confident. Otherwise the mention goes in `unknown_entities` for the review queue. The extractor never invents an id.
7. **`valid_from` / `valid_to`** are the world-time window when known (a contract term, a role change). Null is fine. Recorded time comes from the source note, not the extractor.
8. **`type: other`** is allowed and counted. If more than one fact in five is `other` for a customer, the customer extension needs a new type.
9. **Speaker** is a registry contact id, or a role (`customer`, `our-rep`, `unknown`).

## Signal types, base set (closed list)

| Type | Definition | Example claim |
|---|---|---|
| `complaint` | The customer reports something went wrong | Ashgrove received three late deliveries in August 2026 |
| `request` | The customer asks for something that does not exist yet | Ashgrove asked for a night-mode alarm on the X-series pump |
| `praise` | The customer says something works well | Ashgrove's ICU lead called the onboarding "the smoothest we've had" |
| `risk` | Evidence the relationship or revenue is threatened | Ashgrove's CFO froze new equipment purchases until Q1 2027 |
| `competitor_mention` | A named competitor comes up | Ashgrove is evaluating a Corvane pump for the Springfield ICU |
| `commitment` | Either side promises something | Our rep promised a root-cause summary by 2026-08-28 |
| `decision` | Something was decided | Ashgrove chose to renew the service contract for one year |
| `contact_change` | A person's role, presence or contact info changed | Maria Chen moved from procurement to clinical operations in 2026-08 |
| `pricing` | Price, discount, budget or invoice discussion | Ashgrove expects a 12 percent volume discount on the 2027 renewal |
| `usage` | How the product is used, how much, where | Springfield runs eleven X-series pumps across two ICUs |
| `question` | The customer asked something and did not get an answer | Ashgrove asked whether the pump supports Epic integration; unanswered |
| `other` | Durable fact that fits none of the above | counted; drives schema growth |

Each type in the extraction prompt gets this definition, the example, and one counter-example (what it is not). That checklist is what makes the model look for each kind explicitly.

## Entity kinds (closed)

`account` · `contact` · `product` · `competitor` · `internal_person` · `other`

Internal people are resolved against the customer's own directory (from the CRM user list), so "Sam" becomes `internal:sam-okafor`.

## Actions

An `action` is not a fact. It becomes an `action` note if it survives merge (dedup against open actions on the same entity). `owner_mention` resolves like a speaker. `due` absolute or null.

## The customer extension (produced at onboarding)

```yaml
customer: mercy-medical-supply
schema_version: "0.1"
extra_types:
  - name: clinical_event
    definition: A patient-safety or clinical-workflow event involving our product
    quote_required: true
vocabulary:
  products: [X-series pump, X-series pump v2, "the X"]
  competitors: [Corvane, Brightline]
  jargon: {"IDN": "integrated delivery network", "GPO": "group purchasing organization"}
aliases:
  ashgrove-medical: [Ashgrove Medical, Ashgrove - AMC, Ashgrove Health System Medical Center]
skip_rules:
  - source_type: email
    when: subject matches /^(Out of Office|Automatic reply)/
severity_overrides:
  complaint: {high_if: [delivery, patient, safety, outage]}
```

Onboarding is where this file gets written, tested against a sample of the customer's real sources, and adjusted. That is the startup fee's work product.

## Segmenting long sources

A 25k-token transcript is extracted in segments of roughly 4k tokens with a small overlap. Each segment call gets the running entity list from earlier segments so names stay consistent. Facts are merged per source before the graph merge: same type, same entities, locators within 90 seconds or 2 paragraphs, claim similarity above threshold means one fact, keeping the earliest locator. This costs the same tokens as one big call and recovers the details that get lost in the middle of long inputs.

## What the merge step does with this

For each fact: search the graph for the entities plus claim text. Top hit above threshold means a **compare** call (small model first, escalate if uncertain) with three outcomes: `same` (bump `updated:`, add the new citation), `extends` (append the claim to the existing note), `contradicts` (append, flag, set `valid_to` on the older claim if the new one is dated later). No hit means a new note. Every written claim carries `^[source-slug @ locator]`.

## Versioning

- `schema_version` bumps on any change to types, required fields or rules.
- Source notes record the version that extracted them.
- A rebuild can target "everything extracted under a version below X."
- The bench (`quality-loop.md`) is re-run on every bump.
