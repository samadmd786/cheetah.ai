"""Generate two additional documents for Phase 3.

  data/merger.txt          — a genuinely DIFFERENT contract (M&A purchase
                             agreement). Same rough size as discovery.txt so
                             the KV-cache eviction story is symmetric.

  data/discovery_v3.txt    — a NEAR-DUPLICATE of discovery.txt. We perturb
                             whitespace only — a single extra space after each
                             period and an occasional extra newline. The bytes
                             differ → vllm-mlx's exact-prefix cache will miss.
                             Token sequence is mostly preserved so SimHash
                             should be a very close match (Hamming << 10/64),
                             which is the robustness story in CLAUDE.md §6.

These are inputs to the Phase 3 multi-doc and near-dup pipelines. Run:

    .venv/bin/python scripts/build_additional_docs.py
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import tiktoken


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "data" / "discovery.txt"
OUT_DIFFERENT = REPO_ROOT / "data" / "merger.txt"
OUT_NEAR_DUP = REPO_ROOT / "data" / "discovery_v3.txt"


# ---------- discovery_v3.txt: near-duplicate (whitespace perturbations only)

def build_near_duplicate(source_text: str) -> str:
    """Near-duplicate of the source: whitespace perturbations plus a small
    handful of token-level edits (a few dollar amounts swapped).

    The byte sequence is meaningfully different (vllm-mlx's exact-prefix
    cache misses 100% of the time) and the token sequence has a few changes
    (SimHash distance is small-but-nonzero — the realistic "amended
    contract" scenario rather than the trivial "same doc, different
    whitespace" one).
    """
    text = source_text
    text = text.replace("\t", "    ")
    text = re.sub(r"\.\s", ".  ", text)

    # Surface-level numeric edits — same flavour of money/section numbers,
    # different exact values. These shift a handful of bits in the SimHash
    # without coming close to the near-dup threshold.
    substitutions = [
        ("US$485,000,000", "US$492,500,000"),
        ("US$520,000,000", "US$515,750,000"),
        ("US$50,000,000",  "US$55,000,000"),
        ("US$75,000,000",  "US$80,000,000"),
        ("US$10,000,000",  "US$12,500,000"),
        ("eight hundred (800) basis points", "nine hundred (900) basis points"),
        ("seven (7) years", "eight (8) years"),
    ]
    for old, new in substitutions:
        text = text.replace(old, new)

    # Inject an extra blank line every ~30 source newlines.
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for i, line in enumerate(lines):
        out.append(line)
        if (i + 1) % 30 == 0:
            out.append("\n")
    return "".join(out)


# ---------- merger.txt: a genuinely different contract

MERGER_HEADER = """\
CONFIDENTIAL — TRANSACTION COUNSEL EYES ONLY
AGREEMENT AND PLAN OF MERGER
between TIBER TECHNOLOGIES, INC. ("Parent"),
       TIBER MERGER SUB I, INC. ("Merger Sub"),
   and AURELIUS DATA SYSTEMS, INC. ("the Company")
Dated as of February 14, 2026

PURPOSE: This document records the definitive Agreement and Plan of Merger
governing the proposed cash-and-stock acquisition of the Company by Parent
through a forward triangular merger under Section 368(a)(2)(D) of the Code.
Reverse triangular structure was considered and rejected for the reasons set
forth in Schedule 1(c). Bates: TIB-AUR-0000001 — TIB-AUR-0007214.

PARTIES:
  (i)   Tiber Technologies, Inc., a Delaware corporation having its principal
        place of business at 220 W. 7th Street, 18th Floor, Austin, Texas
        78701 ("Parent");
  (ii)  Tiber Merger Sub I, Inc., a Delaware corporation and direct wholly-
        owned subsidiary of Parent ("Merger Sub"), formed solely for the
        purpose of effecting the transactions contemplated by this Agreement;
  (iii) Aurelius Data Systems, Inc., a Delaware corporation having its
        principal place of business at 1900 Pike Place, Suite 1000, Seattle,
        Washington 98101 (the "Company").

CONSIDERATION: Aggregate consideration payable to Company stockholders shall
consist of (a) US$1,275,000,000 in cash and (b) 14,832,917 shares of Parent
common stock (NYSE: TIB), with per-share consideration calculated as set
forth in Section 2.05 (the "Merger Consideration"). Closing is conditioned
on, among other things, HSR clearance and CFIUS no-action letter.
"""


MERGER_ARTICLES: list[tuple[str, list[str]]] = [
    (
        "ARTICLE I — THE MERGER",
        [
            "Section 1.01. The Merger. Upon the terms and subject to the conditions set forth in this Agreement, and in accordance with the Delaware General Corporation Law (the \"DGCL\"), at the Effective Time, Merger Sub shall be merged with and into the Company (the \"Merger\"), with the Company continuing as the surviving corporation (the \"Surviving Corporation\") and as a direct wholly-owned subsidiary of Parent.",
            "Section 1.02. Effective Time. Subject to the provisions of this Agreement, at the Closing the parties shall cause the Merger to be consummated by filing a certificate of merger with the Secretary of State of the State of Delaware in such form as is required by, and executed in accordance with, the relevant provisions of the DGCL. The Merger shall become effective at such time as the certificate of merger is duly filed with the Delaware Secretary of State, or at such later time as Parent and the Company shall agree and specify in the certificate of merger (the time at which the Merger becomes effective being the \"Effective Time\").",
            "Section 1.03. Closing. The closing of the Merger (the \"Closing\") shall take place at 10:00 a.m. New York City time on a date to be specified by Parent and the Company, which shall be no later than the second Business Day after the satisfaction or waiver (subject to applicable Law) of the conditions set forth in Article VII (other than those conditions that by their nature are to be satisfied at the Closing, but subject to the satisfaction or waiver of those conditions at such time), at the offices of Cravath, Swaine & Moore LLP, 825 Eighth Avenue, New York, NY 10019, unless another date, time or place is agreed to in writing by Parent and the Company.",
            "Section 1.04. Effects of the Merger. The Merger shall have the effects set forth in this Agreement and in the applicable provisions of the DGCL. Without limiting the generality of the foregoing, and subject thereto, at the Effective Time, all the property, rights, privileges, powers and franchises of the Company and Merger Sub shall vest in the Surviving Corporation, and all debts, liabilities and duties of the Company and Merger Sub shall become the debts, liabilities and duties of the Surviving Corporation.",
            "Section 1.05. Certificate of Incorporation and Bylaws. At the Effective Time, the certificate of incorporation of Merger Sub, as in effect immediately prior to the Effective Time, shall be the certificate of incorporation of the Surviving Corporation until thereafter changed or amended as provided therein or by applicable Law. At the Effective Time, the bylaws of Merger Sub, as in effect immediately prior to the Effective Time, shall be the bylaws of the Surviving Corporation until thereafter changed or amended as provided therein or by applicable Law.",
            "Section 1.06. Directors and Officers. The directors of Merger Sub immediately prior to the Effective Time shall be the initial directors of the Surviving Corporation, each to hold office in accordance with the certificate of incorporation and bylaws of the Surviving Corporation until their respective successors are duly elected or appointed and qualified. The officers of the Company immediately prior to the Effective Time shall be the initial officers of the Surviving Corporation, each to hold office in accordance with the bylaws of the Surviving Corporation.",
        ],
    ),
    (
        "ARTICLE II — CONVERSION OF SECURITIES",
        [
            "Section 2.01. Conversion of Company Common Stock. At the Effective Time, each share of common stock, par value $0.001 per share, of the Company (\"Company Common Stock\") issued and outstanding immediately prior to the Effective Time (other than shares to be cancelled pursuant to Section 2.02 and any Dissenting Shares) shall be converted automatically into the right to receive the Per Share Merger Consideration, without interest.",
            "Section 2.02. Cancellation of Treasury and Parent-Owned Shares. At the Effective Time, each share of Company Common Stock held in the Company's treasury or owned by Parent, Merger Sub, or any direct or indirect wholly-owned subsidiary of Parent or the Company immediately prior to the Effective Time shall be cancelled and retired and shall cease to exist, and no consideration shall be paid or payable in respect thereof.",
            "Section 2.03. Conversion of Merger Sub Common Stock. At the Effective Time, each share of common stock, par value $0.01 per share, of Merger Sub issued and outstanding immediately prior to the Effective Time shall be converted into and become one validly issued, fully paid and non-assessable share of common stock, par value $0.01 per share, of the Surviving Corporation.",
            "Section 2.04. Stock Option and Equity Award Treatment. (a) Each outstanding option to purchase Company Common Stock (each, a \"Company Option\") that is vested and exercisable as of immediately prior to the Effective Time shall, by virtue of the Merger and without any action on the part of the holder thereof, be cancelled and converted into the right to receive a cash payment equal to (i) the excess, if any, of the Per Share Cash Consideration over the per-share exercise price of such Company Option, multiplied by (ii) the number of shares of Company Common Stock subject thereto. (b) Each unvested Company Option outstanding immediately prior to the Effective Time shall be assumed by Parent and converted into an option to acquire shares of Parent common stock, with adjustments to the number of shares and exercise price determined in accordance with Section 409A of the Code.",
            "Section 2.05. Per Share Merger Consideration. The \"Per Share Merger Consideration\" means (a) US$36.50 in cash (the \"Per Share Cash Consideration\"), plus (b) 0.4250 shares of Parent common stock (the \"Per Share Stock Consideration\"), in each case without interest and subject to any required tax withholding. The Per Share Stock Consideration is subject to adjustment in the event of any stock split, stock dividend, recapitalization, reclassification, or similar transaction affecting Parent common stock between the date hereof and the Effective Time.",
            "Section 2.06. Dissenting Shares. Notwithstanding any other provision of this Agreement to the contrary, shares of Company Common Stock that are issued and outstanding immediately prior to the Effective Time and that are held by stockholders who have not voted in favor of the adoption of this Agreement and who have properly exercised and perfected their appraisal rights pursuant to Section 262 of the DGCL (\"Dissenting Shares\") shall not be converted into the right to receive the Per Share Merger Consideration.",
        ],
    ),
    (
        "ARTICLE III — REPRESENTATIONS AND WARRANTIES OF THE COMPANY",
        [
            "Section 3.01. Organization, Standing and Power. The Company (a) is a corporation duly organized, validly existing and in good standing under the Laws of the State of Delaware, (b) has all requisite corporate power and authority to own, lease and operate its properties and to carry on its business as now being conducted, and (c) is duly qualified to do business and is in good standing as a foreign corporation in each jurisdiction in which the nature of its business or the ownership or leasing of its properties makes such qualification necessary, except where the failure to be so qualified or in good standing would not, individually or in the aggregate, reasonably be expected to have a Material Adverse Effect.",
            "Section 3.02. Subsidiaries. Section 3.02 of the Company Disclosure Letter sets forth a true and complete list of all subsidiaries of the Company, together with the jurisdiction of organization of each such subsidiary and the percentage of each such subsidiary's outstanding capital stock or other equity interests owned by the Company or another subsidiary of the Company. All the outstanding shares of capital stock or other equity interests of each subsidiary of the Company have been validly issued and are fully paid, nonassessable and owned free and clear of all Liens.",
            "Section 3.03. Capital Structure. The authorized capital stock of the Company consists of (a) 200,000,000 shares of Company Common Stock and (b) 25,000,000 shares of preferred stock, par value $0.001 per share (\"Company Preferred Stock\"). At the close of business on the Business Day immediately preceding the date hereof, (i) 67,418,022 shares of Company Common Stock were issued and outstanding, (ii) no shares of Company Preferred Stock were issued or outstanding, and (iii) Company Options to purchase an aggregate of 4,892,144 shares of Company Common Stock were issued and outstanding.",
            "Section 3.04. Corporate Authority; Voting Requirements. (a) The Company has all requisite corporate power and authority to execute and deliver this Agreement, to perform its obligations hereunder and to consummate the transactions contemplated hereby, subject to receipt of the Company Stockholder Approval. The execution, delivery and performance of this Agreement by the Company and the consummation by the Company of the transactions contemplated hereby have been duly and validly authorized by all necessary corporate action on the part of the Company, subject to receipt of the Company Stockholder Approval. (b) The affirmative vote of the holders of a majority of the outstanding shares of Company Common Stock entitled to vote on the adoption of this Agreement is the only vote of the holders of any class or series of capital stock of the Company necessary to adopt this Agreement and approve the transactions contemplated hereby (the \"Company Stockholder Approval\").",
            "Section 3.05. No Conflict; Required Filings and Consents. Subject to receipt of the Company Stockholder Approval and the consents and filings described in clauses (i) through (vii) of Section 3.05(b), the execution, delivery and performance of this Agreement by the Company and the consummation of the transactions contemplated hereby do not and will not (a) conflict with or violate the certificate of incorporation or bylaws of the Company, (b) conflict with or violate any Law applicable to the Company or any of its subsidiaries or by which any property or asset of the Company or any of its subsidiaries is bound, or (c) result in any breach of, or constitute a default under, any Contract to which the Company or any of its subsidiaries is a party or by which any of their respective properties or assets is bound.",
        ],
    ),
    (
        "ARTICLE IV — REPRESENTATIONS AND WARRANTIES OF PARENT AND MERGER SUB",
        [
            "Section 4.01. Organization, Standing and Power. Each of Parent and Merger Sub is a corporation duly organized, validly existing and in good standing under the Laws of the State of Delaware and has all requisite corporate power and authority to own, lease and operate its properties and to carry on its business as now being conducted.",
            "Section 4.02. Corporate Authority. Each of Parent and Merger Sub has all requisite corporate power and authority to execute and deliver this Agreement, to perform its obligations hereunder and to consummate the transactions contemplated hereby. The execution, delivery and performance of this Agreement by Parent and Merger Sub and the consummation by Parent and Merger Sub of the transactions contemplated hereby have been duly and validly authorized by all necessary corporate action on the part of Parent and Merger Sub.",
            "Section 4.03. Capitalization of Merger Sub. The authorized capital stock of Merger Sub consists of 1,000 shares of common stock, par value $0.01 per share, all of which are validly issued and outstanding. All the issued and outstanding capital stock of Merger Sub is, and immediately prior to the Effective Time will be, owned by Parent. Merger Sub has not conducted any business prior to the date hereof and has no, and immediately prior to the Effective Time will have no, assets, liabilities or obligations of any nature other than those incident to its formation and pursuant to this Agreement.",
            "Section 4.04. Available Funds; Financing. At the Effective Time, Parent will have, and will cause Merger Sub to have, sufficient cash and other immediately available funds, together with proceeds from the Debt Financing, to permit Parent and Merger Sub to (a) pay the aggregate Per Share Cash Consideration, (b) pay all amounts payable under the Company Stock Plans and pursuant to the Company Equity Awards, and (c) pay all fees and expenses payable by Parent and Merger Sub in connection with the Merger and the Financing.",
            "Section 4.05. No Conflict; Required Filings and Consents. The execution, delivery and performance of this Agreement by Parent and Merger Sub and the consummation of the transactions contemplated hereby do not and will not (a) conflict with or violate the certificate of incorporation or bylaws of Parent or Merger Sub, (b) conflict with or violate any Law applicable to Parent or Merger Sub, or (c) result in any breach of, or constitute a default under, any material Contract to which Parent or Merger Sub is a party.",
        ],
    ),
    (
        "ARTICLE V — COVENANTS",
        [
            "Section 5.01. Conduct of Business of the Company. The Company covenants and agrees that, from the date of this Agreement until the earlier of the Effective Time and the termination of this Agreement, except (a) as set forth in Section 5.01 of the Company Disclosure Letter, (b) as required by applicable Law, (c) as expressly required or permitted by this Agreement, or (d) with the prior written consent of Parent (which consent shall not be unreasonably withheld, conditioned or delayed), the Company shall, and shall cause each of its subsidiaries to, conduct its business in the ordinary course consistent with past practice and use commercially reasonable efforts to preserve intact its business organization and goodwill.",
            "Section 5.02. No Solicitation. (a) The Company agrees that it shall not, and shall cause its subsidiaries and its and their respective Representatives not to, directly or indirectly, (i) solicit, initiate, knowingly encourage or knowingly facilitate any Takeover Proposal or any inquiry, proposal or offer that would reasonably be expected to lead to a Takeover Proposal, (ii) furnish to any Person (other than Parent, Merger Sub or any designee of Parent or Merger Sub) any information relating to the Company or any of its subsidiaries in connection with any Takeover Proposal, or (iii) enter into any letter of intent, memorandum of understanding, agreement in principle or other similar Contract with respect to any Takeover Proposal.",
            "Section 5.03. Stockholder Meeting; Proxy Statement. As promptly as practicable following the date of this Agreement, the Company shall (a) prepare and file with the SEC a preliminary proxy statement relating to the Company Stockholder Meeting (the \"Proxy Statement\"), (b) use its reasonable best efforts to clear comments from the SEC staff on the preliminary Proxy Statement, (c) cause the definitive Proxy Statement to be mailed to the stockholders of the Company, and (d) duly call, give notice of, convene and hold the Company Stockholder Meeting for the purpose of obtaining the Company Stockholder Approval.",
            "Section 5.04. Regulatory Matters. (a) Parent and the Company shall each use their respective reasonable best efforts to (i) take, or cause to be taken, all actions, and to do, or cause to be done, all things necessary, proper or advisable under applicable Law to consummate and make effective the transactions contemplated by this Agreement as promptly as practicable, including preparing and filing as promptly as practicable all documentation to effect all necessary applications, notices, petitions, filings and other documents, and to obtain as promptly as practicable all consents, clearances, waivers, licenses, orders, registrations, approvals, permits and authorizations necessary or advisable to be obtained from any Governmental Entity in order to consummate the transactions contemplated by this Agreement.",
            "Section 5.05. Indemnification of Directors and Officers. Parent agrees that all rights to exculpation, indemnification and advancement of expenses now existing in favor of each Person who is now, or has been at any time prior to the date hereof or who becomes prior to the Effective Time, an officer or director of the Company or any of its subsidiaries as provided in the certificate of incorporation or bylaws of the Company or such subsidiary, or in any indemnification agreement, shall be assumed by the Surviving Corporation in the Merger, without further action, as of the Effective Time and shall survive the Merger.",
        ],
    ),
    (
        "ARTICLE VI — CONDITIONS PRECEDENT",
        [
            "Section 6.01. Conditions to Each Party's Obligation to Effect the Merger. The respective obligations of each party to effect the Merger shall be subject to the satisfaction or waiver (where permissible) at or prior to the Effective Time of the following conditions: (a) the Company Stockholder Approval shall have been obtained; (b) no Governmental Entity of competent jurisdiction shall have enacted, issued, promulgated, enforced or entered any Law or Order (whether temporary, preliminary or permanent) that is in effect and prohibits or makes illegal the consummation of the Merger; and (c) any waiting period applicable to the consummation of the Merger under the HSR Act shall have expired or been terminated.",
            "Section 6.02. Conditions to Parent's and Merger Sub's Obligations. The obligations of Parent and Merger Sub to effect the Merger shall be subject to the satisfaction or waiver by Parent on or prior to the Effective Time of the following additional conditions: (a) each of the representations and warranties of the Company set forth in this Agreement shall be true and correct as of the date of this Agreement and as of the Effective Time as if made as of the Effective Time (except to the extent such representations and warranties expressly relate to an earlier date, in which case as of such earlier date); and (b) the Company shall have performed in all material respects all obligations required to be performed by it under this Agreement at or prior to the Effective Time.",
            "Section 6.03. Conditions to the Company's Obligations. The obligations of the Company to effect the Merger shall be subject to the satisfaction or waiver by the Company on or prior to the Effective Time of the following additional conditions: (a) each of the representations and warranties of Parent and Merger Sub set forth in this Agreement shall be true and correct in all material respects as of the date of this Agreement and as of the Effective Time as if made as of the Effective Time; and (b) Parent and Merger Sub shall have performed in all material respects all obligations required to be performed by them under this Agreement at or prior to the Effective Time.",
        ],
    ),
    (
        "ARTICLE VII — TERMINATION",
        [
            "Section 7.01. Termination by Mutual Consent. This Agreement may be terminated and the Merger may be abandoned at any time prior to the Effective Time, whether before or after the receipt of the Company Stockholder Approval, by mutual written consent of Parent and the Company.",
            "Section 7.02. Termination by Either Parent or the Company. This Agreement may be terminated and the Merger may be abandoned at any time prior to the Effective Time, whether before or after the receipt of the Company Stockholder Approval, by either Parent or the Company if: (a) the Merger shall not have been consummated on or before November 14, 2026 (the \"End Date\"); provided, however, that the right to terminate this Agreement pursuant to this Section 7.02(a) shall not be available to any party whose breach of any provision of this Agreement has been the principal cause of, or resulted in, the failure of the Merger to be consummated by such time; (b) any Governmental Entity of competent jurisdiction shall have issued an Order or taken any other action permanently restraining, enjoining or otherwise prohibiting the consummation of the Merger, and such Order or other action shall have become final and non-appealable; or (c) the Company Stockholder Approval shall not have been obtained at the Company Stockholder Meeting (or any adjournment or postponement thereof at which the vote on the adoption of this Agreement was taken).",
            "Section 7.03. Termination Fee. (a) In the event that this Agreement is terminated by Parent or the Company pursuant to specified circumstances enumerated in Section 7.03(b), the Company shall pay or cause to be paid to Parent the Termination Fee in immediately available funds within two (2) Business Days of such termination. (b) The \"Termination Fee\" shall mean an amount equal to US$48,500,000.",
        ],
    ),
]


MERGER_SCHEDULE_TITLES = [
    "Company Disclosure Letter Cross-Reference Index",
    "Schedule of Permitted Capital Expenditures",
    "Identified Material Contracts",
    "Pending and Threatened Litigation Inventory",
    "Tax Sharing Arrangements",
    "Intellectual Property Schedule",
    "List of Key Employees and Retention Targets",
    "Real Property Owned, Leased, or Subleased",
    "Outstanding Indebtedness for Borrowed Money",
    "Affiliate Transactions Requiring Disclosure",
    "Environmental Permits and Reports",
    "Insurance Policy Inventory",
]


def merger_schedule(num: int, title: str) -> str:
    lines: list[str] = []
    for i in range(1, 26):
        lines.append(
            f"  M-{num}.{i:02d}  Item {i} of Schedule M-{num} ({title}). Subject "
            f"to the disclosure standards of Section 3.01 and the regulatory "
            f"covenants of Section 5.04, this item shall be reviewed at each "
            f"quarterly meeting of the Joint Steering Committee. Cross-refs: "
            f"Article II §2.05, Article III §3.04, Article V §5.02, Article VII §7.03."
        )
    body = "\n".join(lines)
    return f"\nSCHEDULE M-{num} — {title}\n{'-' * 60}\n{body}\n"


def build_merger_doc(target_tokens: int) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    parts: list[str] = [MERGER_HEADER]
    for title, sections in MERGER_ARTICLES:
        parts.append(
            f"\n────────────────────────────────────────────────────────────────────────────────\n"
            f"{title}\n"
            f"────────────────────────────────────────────────────────────────────────────────\n"
        )
        parts.extend(s + "\n" for s in sections)

    n = 1
    while True:
        text = "".join(parts)
        if len(enc.encode(text)) >= target_tokens:
            break
        title = MERGER_SCHEDULE_TITLES[(n - 1) % len(MERGER_SCHEDULE_TITLES)]
        parts.append(merger_schedule(n, title))
        n += 1
    return "".join(parts)


# ---------------------------------------------------------------- entry point

def main() -> None:
    random.seed(0)
    enc = tiktoken.get_encoding("cl100k_base")

    src_text = SRC.read_text()
    src_tokens = len(enc.encode(src_text))

    near = build_near_duplicate(src_text)
    OUT_NEAR_DUP.write_text(near)
    near_tokens = len(enc.encode(near))

    merger = build_merger_doc(target_tokens=src_tokens)
    OUT_DIFFERENT.write_text(merger)
    merger_tokens = len(enc.encode(merger))

    print(
        f"discovery.txt        : {len(src_text):>7,d} chars, ~{src_tokens:>6,d} tokens (source)"
    )
    print(
        f"discovery_v3.txt     : {len(near):>7,d} chars, ~{near_tokens:>6,d} tokens "
        f"(near-duplicate of discovery.txt; whitespace perturbed)"
    )
    print(
        f"merger.txt           : {len(merger):>7,d} chars, ~{merger_tokens:>6,d} tokens "
        f"(distinct M&A contract; similar size)"
    )


if __name__ == "__main__":
    main()
