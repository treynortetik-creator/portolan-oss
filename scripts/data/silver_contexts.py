"""Company-context bodies for the public silver-labeling corpora
(`01-spec/training-data-2026-09-19.md`, "Next step"; silver-labels build
instructions item 1: "write one short neutral context per corpus (who the
'vendor' is)... since the extractor prompt takes a company context").
Two more were added 2026-09-22 (`cfpb`, `saas_sales`) -- see
`var/data/MANIFEST.md` sections 6-7.

None of these datasets is Halvard Medical's own text -- each is a real
public corpus in an unrelated domain (medical-device complaints,
design-team meetings, 2000s energy-trading email, Ubuntu IRC support,
CFPB consumer-finance complaints, synthetic B2B SaaS sales dialogue) --
so `portolan.context.HALVARD_MEDICAL_CONTEXT_BODY` does not apply here.
Each body below is a short, neutral, factually-grounded note (same shape
`portolan.extract.prompt.build_system_prompt`'s `company_context` param
expects: plain text, no "## Company context" heading of its own --
`render_company_context_block` adds that) telling the extractor who the
"vendor" side of each corpus is, so fact rule 1 ("state ... whether a
signal helps or hurts the *company*") has a company to reason about, the
same way it would for a real tenant's own context note.

Kept as plain data (a `dict[str, str]`) rather than a `CompanyContext`
model or a database row: `portolan.context.load_company_context` reads
from Postgres, and this silver-labeling pipeline has no tenant database
-- these bodies are passed straight into
`portolan.extract.prompt.build_system_prompt(company_context=...)`, which
only ever wants a plain string.
"""

from __future__ import annotations

__all__ = ["CORPUS_CONTEXTS", "context_for_corpus"]


CORPUS_CONTEXTS: dict[str, str] = {
    "fda_maude": (
        "You are reading medical device complaint narratives from the FDA's public "
        "MAUDE adverse-event database. The 'vendor' side of these records is the "
        "device manufacturer's own support and quality desk: the team that receives "
        "a report of a device malfunction or patient-safety event from a hospital, "
        "clinician, or patient, investigates it, and where the record has a "
        "follow-up narrative, replies with its own findings. There is no single "
        "named company -- MAUDE aggregates reports across many device makers -- so "
        "treat each record's own manufacturer as that record's vendor. What matters: "
        "a reported device malfunction or adverse patient outcome (complaint/risk), "
        "a manufacturer's corrective or follow-up action (commitment), and any "
        "patient-safety risk language."
    ),
    "ami": (
        "You are reading transcripts from the AMI Meeting Corpus: real and "
        "role-played business meetings at a fictional consumer-electronics product "
        "team (the corpus's own 'remote control design' scenario) working through "
        "a product's design, cost, and user-requirements decisions across project "
        "phases. The 'vendor' side is this product team itself -- treat the "
        "speakers as internal staff (marketing, industrial design, user interface, "
        "project management) discussing their own product, not a customer talking "
        "to an outside company. What matters: design decisions, commitments to "
        "follow up on a requirement, disagreements or risks about the product's "
        "direction, and any competitor or market comparison a speaker raises."
    ),
    "enron": (
        "You are reading internal email from Enron, a Houston-based energy trading "
        "and utilities company (this corpus is the public record released after "
        "Enron's 2001 collapse, covering roughly 2000-2002). The 'vendor' side is "
        "Enron itself -- treat the email's senders and recipients as Enron staff "
        "(trading desks, scheduling, HR, legal, executive) corresponding about "
        "Enron's own business, not a customer writing in to an outside vendor. "
        "What matters: business decisions and commitments made over email, "
        "complaints or risks raised about a deal, counterparty, or internal "
        "process, and any pricing or contract terms discussed."
    ),
    "ubuntu_irc": (
        "You are reading messages from the #ubuntu IRC channel, a public volunteer "
        "technical-support chat for the Ubuntu Linux operating system. The "
        "'vendor' side is Canonical/the Ubuntu project's own support channel -- "
        "treat the channel's regular helpers and any bot as the vendor's support "
        "desk, and a user asking for help as the customer. What matters: a "
        "reported problem or bug (complaint), a request for help or a feature "
        "(request), a volunteer's fix or workaround (commitment), and any praise "
        "or frustration a user expresses about Ubuntu."
    ),
    "cfpb": (
        "You are reading consumer complaints from the CFPB's public Consumer "
        "Complaint Database, each with the consumer's own narrative and, when "
        "present, the company's response. The 'vendor' side is a consumer "
        "financial company's complaint desk -- each record's own named company "
        "(a bank, lender, credit reporting agency, or debt collector) receiving "
        "and responding to a complaint about its own product or service. What "
        "matters: the consumer's reported problem or financial harm (complaint), "
        "any risk language, and the company's own response or corrective action "
        "(commitment) when the record has one."
    ),
    "saas_sales": (
        "You are reading synthetic sales conversations between a B2B software "
        "vendor's sales team and a prospective buyer. The 'vendor' side is the "
        "sales representative selling a SaaS product; the buyer is a named "
        "company evaluating it. What matters: pricing or contract terms "
        "discussed (pricing), any competitor the buyer mentions "
        "(competitor_mention), a commitment either side makes (commitment), a "
        "decision or next step (decision), and any risk or objection the buyer "
        "raises about adopting the product (risk)."
    ),
}


def context_for_corpus(corpus: str) -> str:
    """The company-context body for `corpus` (one of `CORPUS_CONTEXTS`'
    keys). Raises `KeyError` with the available names on an unknown
    corpus -- never silently returns an empty/generic context, since a
    missing context would change the fact rule 1 framing the extractor
    reasons under without anyone noticing."""
    try:
        return CORPUS_CONTEXTS[corpus]
    except KeyError as exc:
        raise KeyError(
            f"no silver-label company context for corpus {corpus!r}; "
            f"known corpora: {sorted(CORPUS_CONTEXTS)}"
        ) from exc
