"""Generate a synthetic litigation / discovery contract of a target token length.

The output (data/discovery.txt) is the heavy document used by every agent in the
Phase 1 pipeline. We want it large enough (~25-30k tokens) that re-prefilling it
three times in BEFORE mode is visibly painful, and a single prefill in AFTER mode
is a visibly large win.

The text is intentionally structured with:
  - a few "conflict of interest" mentions (for Screener),
  - a few "financial liability" clauses (for Analyst),
  - lots of plausible boilerplate between them (so the doc is realistically large).
"""
from __future__ import annotations

from pathlib import Path

import tiktoken

TARGET_TOKENS = 12000  # ~10-30k window per CLAUDE.md; start at 12k, can bump later
OUT = Path(__file__).resolve().parents[1] / "data" / "discovery.txt"


HEADER = """\
CONFIDENTIAL DISCOVERY MATERIAL — CASE NO. 2026-CV-118402
IN RE: MERIDIAN-HALSTEAD HOLDINGS, LLC v. ARGENT CAPITAL PARTNERS, LP
United States District Court for the Northern District of Illinois, Eastern Division

DOCUMENT TITLE: Master Services and Joint Operating Agreement (the "Agreement"),
together with all schedules, exhibits, side letters, and amendments incorporated
herein by reference. Bates range MHH-ARG-0000001 through MHH-ARG-0004812.

PARTIES:
  (i)  Meridian-Halstead Holdings, LLC, a Delaware limited liability company having
       its principal place of business at 500 W. Madison Street, Suite 3400,
       Chicago, Illinois 60661 ("Meridian");
  (ii) Argent Capital Partners, LP, a Cayman Islands exempted limited partnership
       acting through its general partner Argent GP Ltd., with its registered
       office at PO Box 309, Ugland House, Grand Cayman KY1-1104 ("Argent");
  (iii) for purposes of Articles VII and XII only, Halstead Trust Services, N.A.,
       solely in its capacity as collateral agent ("Collateral Agent").

EFFECTIVE DATE: January 3, 2024 (the "Effective Date").

RECITALS:
  WHEREAS, Meridian and Argent desire to establish a joint operating framework for
the origination, syndication, and servicing of structured credit obligations
backed by middle-market commercial assets;
  WHEREAS, the parties acknowledge that certain principals of Argent previously
held advisory roles with affiliates of Meridian and that such relationships may
give rise to conflict of interest considerations addressed in Article IX;
  WHEREAS, Meridian shall contribute the Initial Capital Commitment of
US$485,000,000 and Argent shall contribute origination capability and a portfolio
of qualified assets having an aggregate book value of not less than
US$520,000,000;
  NOW, THEREFORE, in consideration of the mutual covenants set forth herein, and
for other good and valuable consideration the receipt and sufficiency of which
are hereby acknowledged, the parties agree as follows.
"""


ARTICLES: list[tuple[str, list[str]]] = [
    (
        "ARTICLE I — DEFINITIONS",
        [
            'Section 1.01. "Affiliate" means, with respect to any Person, any other Person directly or indirectly Controlling, Controlled by, or under common Control with such Person, where "Control" means the possession, directly or indirectly, of the power to direct or cause the direction of the management and policies of a Person, whether through the ownership of voting securities, by contract, or otherwise.',
            'Section 1.02. "Available Cash" means, as of any date of determination, all cash and cash equivalents of the Joint Venture, less (a) reserves established by the Operating Committee in accordance with prudent commercial practice, (b) amounts required to satisfy obligations then due and payable, and (c) the Working Capital Floor of US$15,000,000.',
            'Section 1.03. "Business Day" means any day other than a Saturday, Sunday, or day on which commercial banks in the City of New York or the City of Chicago are authorized or required by Law to close.',
            'Section 1.04. "Capital Account" has the meaning ascribed thereto in Section 4.03(a) and shall be maintained in accordance with the principles set forth in Treasury Regulation §1.704-1(b)(2)(iv).',
            'Section 1.05. "Change of Control" means, with respect to any Party, (a) the acquisition by any Person or group (within the meaning of Section 13(d)(3) of the Exchange Act) of more than fifty percent (50%) of the voting securities of such Party or of any direct or indirect parent thereof; (b) the sale, lease, or transfer of all or substantially all of the assets of such Party; or (c) the consummation of any merger, consolidation, statutory share exchange, or similar transaction following which the equityholders of such Party immediately prior thereto cease to hold a majority of the voting securities of the surviving entity.',
            'Section 1.06. "Confidential Information" means all non-public information, in whatever form or medium, concerning the business, operations, finances, assets, liabilities, strategy, customers, vendors, employees, or trade secrets of a Party or its Affiliates, including but not limited to financial models, deal pipelines, underwriting standards, pricing methodologies, and any information that a reasonable person in the receiving Party\'s position would understand to be confidential.',
            'Section 1.07. "Eligible Asset" means a commercial loan, lease, receivable, or other financial obligation that satisfies the eligibility criteria set forth on Schedule 1.07, as the same may be amended from time to time by unanimous written consent of the Operating Committee.',
            'Section 1.08. "GAAP" means generally accepted accounting principles in the United States, consistently applied, as in effect from time to time.',
            'Section 1.09. "Indebtedness" of any Person means, without duplication, (a) indebtedness for borrowed money, (b) obligations evidenced by bonds, debentures, notes, or similar instruments, (c) reimbursement obligations in respect of letters of credit, (d) capitalized lease obligations, (e) guarantees of any of the foregoing, and (f) any obligation to pay the deferred purchase price of property or services, other than trade payables incurred in the ordinary course of business.',
            'Section 1.10. "Liability Cap" has the meaning ascribed thereto in Section 11.04 and represents the maximum aggregate financial liability of either Party for breach of this Agreement, subject only to the carve-outs enumerated in Section 11.05.',
            'Section 1.11. "Material Adverse Effect" means any event, change, occurrence, or development that, individually or in the aggregate, has had or could reasonably be expected to have a material adverse effect on (a) the business, assets, results of operations, or financial condition of the Joint Venture, taken as a whole, or (b) the ability of any Party to perform its material obligations under this Agreement.',
            'Section 1.12. "Operating Committee" means the governance body established pursuant to Article III, comprised of two (2) representatives appointed by Meridian and two (2) representatives appointed by Argent, with such additional non-voting observers as the Parties may designate.',
            'Section 1.13. "Permitted Transferee" means, with respect to any Party, (a) any wholly-owned Affiliate of such Party, (b) any successor entity by way of merger, consolidation, or reorganization, and (c) any transferee approved in writing by the non-transferring Party, such approval not to be unreasonably withheld, conditioned, or delayed.',
            'Section 1.14. "Pro Rata Share" means, with respect to any Party as of any date of determination, the percentage equal to the quotient of (x) such Party\'s Capital Account balance divided by (y) the aggregate Capital Account balances of all Parties, expressed as a percentage and rounded to the nearest one-hundredth of one percent.',
            'Section 1.15. "Termination Event" has the meaning ascribed thereto in Section 12.02 and includes, without limitation, the events of default, insolvency, regulatory disqualification, and material breach more particularly described therein.',
        ],
    ),
    (
        "ARTICLE II — FORMATION AND PURPOSE OF THE JOINT VENTURE",
        [
            "Section 2.01. Formation. The Parties hereby form a joint venture in the form of a Delaware limited liability company to be known as Meridian-Argent Structured Credit Partners I, LLC (the \"Joint Venture\"), pursuant to and in accordance with the Delaware Limited Liability Company Act.",
            "Section 2.02. Purpose. The purpose of the Joint Venture shall be to (a) originate, underwrite, and acquire Eligible Assets; (b) structure and syndicate securitization vehicles backed by such Eligible Assets; (c) provide ongoing servicing, administration, and reporting in respect of such vehicles; and (d) engage in such other activities as the Operating Committee may from time to time approve by Supermajority Vote.",
            "Section 2.03. Term. The Joint Venture shall commence on the Effective Date and shall continue in existence for an initial term of seven (7) years, with automatic successive one-year renewals unless terminated in accordance with Article XII.",
            "Section 2.04. Principal Office. The principal office of the Joint Venture shall be located at 500 W. Madison Street, Suite 3400, Chicago, Illinois 60661, or such other location as the Operating Committee may designate.",
            "Section 2.05. Registered Agent. The registered agent for service of process in the State of Delaware shall be Corporation Trust Center, 1209 Orange Street, Wilmington, Delaware 19801.",
            "Section 2.06. Conflict of Interest Disclosure — Initial. The Parties acknowledge and disclose that as of the Effective Date, three (3) principals of Argent — namely, Mr. Daniel R. Voss, Ms. Patricia Yelena Cho, and Mr. Marcus T. Albright — previously held senior advisory positions with Meridian Asset Advisors, LLC, an Affiliate of Meridian, during the period from 2019 through 2023. The Parties have implemented the information barriers described in Section 9.02 to mitigate any conflict of interest that may arise from such prior relationships.",
            "Section 2.07. Regulatory Status. Each Party represents that it has obtained, and shall maintain throughout the term of this Agreement, all licenses, registrations, and qualifications necessary to perform its obligations hereunder, including without limitation registration as an investment adviser under the Investment Advisers Act of 1940 to the extent applicable.",
            "Section 2.08. Tax Classification. The Parties intend that the Joint Venture be treated as a partnership for U.S. federal income tax purposes and shall not elect to be classified as a corporation under Treasury Regulation §301.7701-3.",
        ],
    ),
    (
        "ARTICLE III — GOVERNANCE",
        [
            "Section 3.01. Operating Committee. The business and affairs of the Joint Venture shall be managed by the Operating Committee, which shall consist of four (4) voting members: two (2) appointed by Meridian (the \"Meridian Representatives\") and two (2) appointed by Argent (the \"Argent Representatives\"). Each Party may also designate up to two (2) non-voting observers.",
            "Section 3.02. Meetings. The Operating Committee shall meet not less than once per calendar quarter and at such additional times as any member may reasonably request upon not less than five (5) Business Days' prior written notice.",
            "Section 3.03. Quorum. A quorum for any meeting of the Operating Committee shall require the presence (in person, by telephone, or by other means of contemporaneous communication) of at least one (1) Meridian Representative and one (1) Argent Representative.",
            "Section 3.04. Voting. Except as otherwise expressly provided herein, all actions of the Operating Committee shall require the affirmative vote of a majority of the voting members present at a duly convened meeting (a \"Majority Vote\").",
            "Section 3.05. Supermajority Matters. Notwithstanding Section 3.04, the following matters shall require the affirmative vote of three (3) of the four (4) voting members (a \"Supermajority Vote\"): (a) any amendment to this Agreement; (b) any acquisition, sale, or disposition of assets having an aggregate value in excess of US$50,000,000; (c) the incurrence of Indebtedness in excess of the Approved Leverage Cap; (d) the admission of any new member; (e) the appointment or removal of any officer; (f) the approval of the annual budget; (g) any related-party transaction (other than transactions in the ordinary course of business consistent with past practice); and (h) the commencement of any litigation, arbitration, or regulatory proceeding having a potential exposure in excess of US$10,000,000.",
            "Section 3.06. Deadlock Resolution. In the event that the Operating Committee is unable to reach the required vote on any matter, the matter shall be referred to the chief executive officers of Meridian and Argent for resolution within ten (10) Business Days. If such officers are unable to resolve the deadlock, either Party may invoke the dispute resolution procedures set forth in Article XIV.",
            "Section 3.07. Conflict of Interest in Voting. Any voting member of the Operating Committee who has a direct or indirect personal financial interest in any matter before the Committee, other than an interest arising solely from such member's ownership of Capital Account interests in proportion to other members, shall disclose such interest and shall recuse themselves from voting on such matter. Failure to disclose such conflict of interest shall constitute a material breach of this Agreement.",
        ],
    ),
    (
        "ARTICLE IV — CAPITAL CONTRIBUTIONS AND CAPITAL ACCOUNTS",
        [
            "Section 4.01. Initial Capital Contributions. On the Effective Date, (a) Meridian shall contribute US$485,000,000 in immediately available funds to the Joint Venture's primary operating account; and (b) Argent shall contribute the Initial Asset Portfolio described on Schedule 4.01, having an aggregate fair market value as of the Effective Date of not less than US$520,000,000 as confirmed by the Independent Valuation Report.",
            "Section 4.02. Additional Capital Calls. The Operating Committee may, by Supermajority Vote, issue capital calls to the Parties in proportion to their respective Pro Rata Shares; provided that no Party shall be required to contribute additional capital exceeding US$100,000,000 in any rolling twelve (12) month period without such Party's prior written consent.",
            "Section 4.03. Capital Accounts. (a) The Joint Venture shall maintain a Capital Account for each Party in accordance with Treasury Regulation §1.704-1(b)(2)(iv). (b) Each Capital Account shall be increased by (i) the amount of cash and the agreed fair market value of property contributed by such Party; and (ii) the amount of income and gain allocated to such Party. (c) Each Capital Account shall be decreased by (i) the amount of cash and the agreed fair market value of property distributed to such Party; and (ii) the amount of loss and deduction allocated to such Party.",
            "Section 4.04. No Interest on Capital. No Party shall be entitled to interest on its Capital Account balance or on any capital contribution.",
            "Section 4.05. Withdrawals. No Party shall have the right to withdraw any portion of its Capital Account except as expressly provided in Article V (Distributions) or Article XII (Termination and Liquidation).",
            "Section 4.06. Financial Liability for Default. In the event that any Party fails to fund a capital call when due, such defaulting Party shall be subject to the financial liability consequences set forth in Section 4.07, including dilution, default interest at the Default Rate, and reimbursement of costs incurred by the non-defaulting Party in funding the shortfall.",
            "Section 4.07. Default Remedies. (a) The defaulting Party's Capital Account shall be reduced by an amount equal to one hundred fifty percent (150%) of the unfunded amount. (b) Default interest at a rate equal to the prime rate published in the Wall Street Journal plus eight hundred (800) basis points per annum shall accrue on the unfunded amount from the date due until paid in full. (c) The non-defaulting Party shall have the right, but not the obligation, to fund the shortfall and receive a corresponding increase in its Capital Account.",
        ],
    ),
    (
        "ARTICLE V — DISTRIBUTIONS",
        [
            "Section 5.01. Distribution Waterfall. Available Cash shall be distributed not less frequently than quarterly in the following order of priority: (a) first, to the payment of accrued and unpaid expenses of the Joint Venture; (b) second, to the establishment of reserves as determined by the Operating Committee; (c) third, to each Party in proportion to such Party's Unreturned Capital, until each Party's Unreturned Capital balance is reduced to zero; (d) fourth, to each Party in proportion to such Party's Pro Rata Share until each Party has received an Internal Rate of Return of eight percent (8%); and (e) thereafter, eighty percent (80%) to the Parties in proportion to their Pro Rata Shares and twenty percent (20%) to Argent as a carried interest (the \"Carried Interest\").",
            "Section 5.02. Tax Distributions. Notwithstanding Section 5.01, the Joint Venture shall use commercially reasonable efforts to distribute to each Party, not later than April 1 of each calendar year, an amount equal to such Party's share of estimated taxable income for the prior year multiplied by the Assumed Tax Rate (currently forty percent (40%)).",
            "Section 5.03. Withholding. The Joint Venture shall withhold from any distribution any amounts required to be withheld under applicable tax Law and shall remit such amounts to the appropriate taxing authorities. Any amounts so withheld shall be treated for all purposes hereof as having been distributed to the affected Party.",
            "Section 5.04. Distributions in Kind. The Operating Committee may, by Supermajority Vote, elect to distribute property in kind in lieu of cash, in which case the value of such property shall be determined by an Independent Appraiser and the distribution shall be made on a pro rata basis to all Parties entitled to receive such distribution.",
            "Section 5.05. Clawback. If, upon the final liquidation of the Joint Venture, Argent has received aggregate Carried Interest distributions in excess of the amount to which Argent would have been entitled had the Carried Interest been calculated on an aggregate basis over the entire term of the Joint Venture, Argent shall return such excess to the Joint Venture for redistribution to the Parties; provided that Argent's aggregate clawback obligation shall not exceed the lesser of (i) the aggregate after-tax Carried Interest distributions actually received by Argent, or (ii) US$75,000,000.",
        ],
    ),
    (
        "ARTICLE VI — ALLOCATIONS OF PROFITS AND LOSSES",
        [
            "Section 6.01. General Allocation. After giving effect to the special allocations set forth in Section 6.02, all items of income, gain, loss, deduction, and credit shall be allocated to the Parties in proportion to their respective Pro Rata Shares.",
            "Section 6.02. Special Allocations. (a) Minimum Gain Chargeback. Notwithstanding any other provision of this Article VI, if there is a net decrease in Joint Venture Minimum Gain (as defined in Treasury Regulation §1.704-2(d)) during any taxable year, each Party shall be specially allocated items of income and gain in accordance with the provisions of Treasury Regulation §1.704-2(f). (b) Member Nonrecourse Deductions. Member Nonrecourse Deductions shall be allocated to the Party that bears the economic risk of loss with respect to the related Member Nonrecourse Debt. (c) Qualified Income Offset. In the event that any Party unexpectedly receives an adjustment, allocation, or distribution described in Treasury Regulation §1.704-1(b)(2)(ii)(d)(4), (5), or (6) that results in a deficit Capital Account balance, items of income and gain shall be specially allocated to such Party in an amount and manner sufficient to eliminate such deficit as quickly as possible.",
            "Section 6.03. Tax Allocations Under Section 704(c). In accordance with Section 704(c) of the Code and the Treasury Regulations thereunder, income, gain, loss, and deduction with respect to any property contributed to the Joint Venture shall, solely for tax purposes, be allocated among the Parties so as to take account of any variation between the adjusted basis of such property and its initial agreed fair market value, using the traditional method described in Treasury Regulation §1.704-3(b).",
            "Section 6.04. Curative Allocations. The allocations set forth in Section 6.02 are intended to comply with the requirements of Treasury Regulation §1.704-1(b) and §1.704-2 and shall be interpreted consistently therewith. Curative allocations shall be made in accordance with Treasury Regulation §1.704-3(c) to the extent necessary to eliminate any economic distortions caused by such regulatory allocations.",
        ],
    ),
    (
        "ARTICLE VII — DEBT FINANCING AND COLLATERAL",
        [
            "Section 7.01. Approved Leverage Cap. The Joint Venture shall not incur Indebtedness in excess of an amount equal to four (4) times the aggregate Capital Account balances of the Parties (the \"Approved Leverage Cap\") without the Supermajority Vote of the Operating Committee.",
            "Section 7.02. Senior Credit Facility. The Joint Venture is authorized to enter into a senior secured credit facility with a syndicate of lenders led by Halstead National Bank in an aggregate commitment amount of up to US$1,500,000,000, on terms substantially consistent with the term sheet attached as Exhibit 7.02.",
            "Section 7.03. Collateral Agent. Halstead Trust Services, N.A. is hereby appointed as Collateral Agent for the benefit of the secured creditors of the Joint Venture, with the rights, duties, and obligations set forth in the Collateral Agency Agreement of even date herewith. The Collateral Agent shall be entitled to indemnification by the Joint Venture for any loss, liability, claim, or expense incurred in the performance of its duties, except to the extent caused by the Collateral Agent's gross negligence or willful misconduct.",
            "Section 7.04. Permitted Liens. The Joint Venture may grant liens on its assets only (a) in favor of the Collateral Agent for the benefit of the secured creditors under the Senior Credit Facility; (b) for taxes not yet due and payable; (c) for liens of mechanics, materialmen, warehousemen, and similar persons arising in the ordinary course of business; and (d) for liens approved by Supermajority Vote of the Operating Committee.",
            "Section 7.05. Limitation on Subsidiary Debt. No subsidiary of the Joint Venture shall incur Indebtedness in excess of US$25,000,000 without the prior approval of the Operating Committee, and the aggregate Indebtedness of all subsidiaries shall not exceed US$200,000,000.",
            "Section 7.06. Cross-Default Provisions. An event of default under the Senior Credit Facility or any other Material Indebtedness of the Joint Venture shall constitute a Termination Event hereunder, entitling either Party to invoke the remedies set forth in Article XII.",
            "Section 7.07. Financial Liability of Parties for Joint Venture Debt. Notwithstanding anything to the contrary in any financing document, no Party shall have personal financial liability for any Indebtedness of the Joint Venture except (a) pursuant to an express written guaranty executed by such Party, or (b) for damages caused by such Party's fraud, willful misconduct, or breach of its representations and warranties under such financing document.",
        ],
    ),
    (
        "ARTICLE VIII — REPRESENTATIONS, WARRANTIES, AND COVENANTS",
        [
            "Section 8.01. Mutual Representations. Each Party represents and warrants to the other that (a) it is duly organized, validly existing, and in good standing under the Laws of its jurisdiction of organization; (b) it has full power and authority to execute, deliver, and perform this Agreement; (c) the execution, delivery, and performance of this Agreement have been duly authorized by all necessary action; (d) this Agreement constitutes its legal, valid, and binding obligation, enforceable against it in accordance with its terms; and (e) the execution, delivery, and performance of this Agreement do not and will not violate any Law, regulation, judgment, order, or agreement to which it is subject.",
            "Section 8.02. Argent-Specific Representations. Argent further represents and warrants that (a) the Initial Asset Portfolio consists solely of Eligible Assets that satisfy the criteria set forth on Schedule 1.07; (b) Argent has good and marketable title to each asset comprising the Initial Asset Portfolio, free and clear of all liens and encumbrances except as disclosed on Schedule 8.02; (c) no asset comprising the Initial Asset Portfolio is the subject of any pending or threatened litigation, arbitration, or regulatory proceeding; and (d) the disclosures regarding conflict of interest set forth in Section 2.06 are true, complete, and accurate in all material respects.",
            "Section 8.03. Meridian-Specific Representations. Meridian further represents and warrants that (a) it has obtained all consents, approvals, and authorizations from its limited partners and other equityholders necessary to consummate the transactions contemplated hereby; (b) the source of the funds comprising its Initial Capital Contribution is lawful and free from any taint of money laundering, bribery, corruption, or other unlawful activity; and (c) Meridian is not a \"prohibited person\" within the meaning of the OFAC sanctions regulations.",
            "Section 8.04. Affirmative Covenants. Each Party covenants and agrees that, during the term of this Agreement, it shall (a) comply in all material respects with all applicable Laws; (b) maintain in effect all licenses, registrations, and qualifications necessary to perform its obligations hereunder; (c) provide the other Party with prompt written notice of any Material Adverse Effect; (d) cooperate in good faith with the other Party in connection with all matters arising under this Agreement; and (e) refrain from taking any action that would reasonably be expected to result in a Termination Event.",
            "Section 8.05. Negative Covenants. Each Party covenants and agrees that, without the prior written consent of the other Party, it shall not (a) directly or indirectly compete with the Joint Venture in the origination of Eligible Assets within the Restricted Territory; (b) solicit for employment any employee of the other Party or its Affiliates; (c) disclose any Confidential Information except as expressly permitted hereunder; (d) take any action that would cause the Joint Venture to fail to qualify as a partnership for U.S. federal income tax purposes; or (e) cause the Joint Venture to enter into any related-party transaction except in compliance with Section 9.04.",
        ],
    ),
    (
        "ARTICLE IX — CONFLICTS OF INTEREST",
        [
            "Section 9.01. General Standard. Each Party acknowledges that, in the course of operating the Joint Venture, situations may arise in which the interests of such Party or its Affiliates are or may appear to be in conflict with the interests of the Joint Venture or the other Party. Each Party shall use good faith efforts to identify, disclose, and manage any such conflict of interest in accordance with the procedures set forth in this Article IX.",
            "Section 9.02. Information Barriers. The Parties shall implement and maintain information barriers (\"Chinese Walls\") to prevent the unauthorized flow of Confidential Information between (a) personnel of either Party assigned to the Joint Venture, on the one hand, and (b) personnel of such Party or its Affiliates engaged in activities that may give rise to a conflict of interest, on the other hand. Such information barriers shall include physical separation, electronic access controls, training, and monitoring procedures consistent with industry best practices.",
            "Section 9.03. Disclosure of Conflicts. Each Party shall promptly disclose to the Operating Committee in writing any actual, potential, or apparent conflict of interest involving such Party, its Affiliates, or any of its personnel. The Operating Committee shall consider such disclosure and shall determine, by Supermajority Vote, the appropriate course of action, which may include recusal, divestiture, or other mitigating measures.",
            "Section 9.04. Related-Party Transactions. No related-party transaction (other than transactions in the ordinary course of business consistent with past practice and on arms'-length terms) shall be entered into by the Joint Venture without the prior approval of the Operating Committee, including the affirmative vote of at least one (1) member appointed by the non-interested Party.",
            "Section 9.05. Updated Conflict of Interest Disclosures. The Parties further disclose the following matters as of the Effective Date: (i) Mr. Daniel R. Voss serves on the board of directors of Halstead Property Partners, LLC, which is an Affiliate of the Collateral Agent; (ii) Ms. Patricia Yelena Cho's spouse is a senior portfolio manager at Meridian Asset Advisors, LLC; and (iii) Argent has previously co-invested with Meridian on three separate transactions totaling approximately US$180,000,000, the details of which are set forth on Schedule 9.05. The Parties agree that, notwithstanding these disclosed relationships, no further action is required at this time, but reserve the right to revisit this determination if circumstances change.",
            "Section 9.06. Personal Trading Restrictions. Each member of the Operating Committee and each employee of the Joint Venture shall comply with the Personal Trading Policy attached as Exhibit 9.06, which prohibits trading in securities of issuers whose assets are held in the Joint Venture's portfolio without the prior written approval of the Compliance Officer.",
            "Section 9.07. Annual Conflicts Certification. On or before March 31 of each calendar year, each member of the Operating Committee and each senior employee of the Joint Venture shall execute and deliver to the Compliance Officer an Annual Conflicts Certification in substantially the form attached as Exhibit 9.07.",
        ],
    ),
    (
        "ARTICLE X — INDEMNIFICATION",
        [
            "Section 10.01. Indemnification by Argent. Argent shall indemnify, defend, and hold harmless Meridian and its Affiliates, and their respective officers, directors, employees, agents, and representatives (the \"Meridian Indemnified Parties\"), from and against any and all losses, liabilities, claims, damages, costs, and expenses (including reasonable attorneys' fees and expenses) arising out of or in connection with (a) any breach by Argent of its representations, warranties, covenants, or agreements set forth in this Agreement; (b) any fraud, willful misconduct, or gross negligence of Argent or its personnel; and (c) any liability arising from the Initial Asset Portfolio prior to the Effective Date.",
            "Section 10.02. Indemnification by Meridian. Meridian shall indemnify, defend, and hold harmless Argent and its Affiliates, and their respective officers, directors, employees, agents, and representatives (the \"Argent Indemnified Parties\"), from and against any and all losses, liabilities, claims, damages, costs, and expenses (including reasonable attorneys' fees and expenses) arising out of or in connection with (a) any breach by Meridian of its representations, warranties, covenants, or agreements set forth in this Agreement; (b) any fraud, willful misconduct, or gross negligence of Meridian or its personnel; and (c) any liability arising from any actions taken or omitted by Meridian or its Affiliates prior to the Effective Date.",
            "Section 10.03. Procedure for Indemnification Claims. Any Person seeking indemnification under this Article X (an \"Indemnified Party\") shall provide the indemnifying Party (the \"Indemnifying Party\") with prompt written notice of any claim for indemnification, which notice shall include a reasonably detailed description of the basis for the claim and the amount of the loss or expense (or, if not then known, a good faith estimate thereof).",
            "Section 10.04. Defense of Third-Party Claims. The Indemnifying Party shall have the right, but not the obligation, to assume the defense of any third-party claim with counsel of its choosing (subject to the reasonable approval of the Indemnified Party). If the Indemnifying Party assumes such defense, the Indemnified Party shall cooperate in good faith with such defense and shall be entitled to participate in such defense at its own expense.",
            "Section 10.05. Settlement. No Indemnifying Party shall settle any third-party claim without the prior written consent of the Indemnified Party (such consent not to be unreasonably withheld), unless such settlement (a) involves only the payment of money damages, (b) includes a complete and unconditional release of the Indemnified Party from all liability, and (c) does not involve any admission of liability or fault on the part of the Indemnified Party.",
            "Section 10.06. Survival. The representations, warranties, covenants, and indemnification obligations set forth herein shall survive the expiration or termination of this Agreement for a period of three (3) years; provided, however, that (a) the indemnification obligations with respect to fraud or willful misconduct shall survive indefinitely; (b) the indemnification obligations with respect to tax matters shall survive for the applicable statute of limitations plus thirty (30) days; and (c) the indemnification obligations with respect to environmental matters shall survive for ten (10) years.",
        ],
    ),
    (
        "ARTICLE XI — LIMITATION OF LIABILITY",
        [
            "Section 11.01. Disclaimer of Consequential Damages. EXCEPT AS EXPRESSLY PROVIDED IN SECTION 11.05, IN NO EVENT SHALL EITHER PARTY BE LIABLE TO THE OTHER PARTY FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL, EXEMPLARY, OR PUNITIVE DAMAGES, INCLUDING WITHOUT LIMITATION LOSS OF PROFITS, LOSS OF BUSINESS OPPORTUNITY, LOSS OF GOODWILL, OR LOSS OF DATA, ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT, WHETHER BASED ON CONTRACT, TORT, STRICT LIABILITY, OR ANY OTHER LEGAL THEORY, EVEN IF SUCH PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.",
            "Section 11.02. Disclaimer of Implied Warranties. EXCEPT FOR THE EXPRESS REPRESENTATIONS AND WARRANTIES SET FORTH IN ARTICLE VIII, NEITHER PARTY MAKES ANY REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED, AND EACH PARTY HEREBY DISCLAIMS ALL OTHER REPRESENTATIONS AND WARRANTIES, INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT.",
            "Section 11.03. Financial Liability for Direct Damages. Subject to the Liability Cap set forth in Section 11.04 and the carve-outs set forth in Section 11.05, each Party's financial liability to the other Party for direct damages arising out of or in connection with this Agreement shall be limited to actual, proven, out-of-pocket damages.",
            "Section 11.04. Liability Cap. EXCEPT AS PROVIDED IN SECTION 11.05, THE AGGREGATE FINANCIAL LIABILITY OF EITHER PARTY TO THE OTHER PARTY UNDER THIS AGREEMENT SHALL NOT EXCEED THE GREATER OF (A) US$50,000,000 OR (B) THE AGGREGATE AMOUNT OF DISTRIBUTIONS RECEIVED BY THE BREACHING PARTY DURING THE TWENTY-FOUR (24) MONTH PERIOD IMMEDIATELY PRECEDING THE BREACH (THE \"LIABILITY CAP\").",
            "Section 11.05. Carve-Outs. The Liability Cap set forth in Section 11.04 and the disclaimer of consequential damages set forth in Section 11.01 shall NOT apply to (a) liability arising from fraud, willful misconduct, or intentional breach of this Agreement; (b) liability arising from a Party's breach of its confidentiality obligations under Article XIII; (c) liability arising from a Party's indemnification obligations under Sections 10.01(c) or 10.02(c) (relating to pre-Effective Date liabilities); (d) liability arising from a Party's breach of the conflict of interest provisions of Article IX where such breach is shown to have caused actual financial harm to the Joint Venture or the other Party; and (e) liability for unpaid taxes, fines, or penalties imposed by a Governmental Authority as a result of a Party's wrongful conduct.",
            "Section 11.06. Insurance. Each Party shall maintain, throughout the term of this Agreement, (a) commercial general liability insurance with a per-occurrence limit of not less than US$10,000,000; (b) errors and omissions insurance with an aggregate limit of not less than US$25,000,000; (c) directors and officers liability insurance with an aggregate limit of not less than US$50,000,000; and (d) such other insurance as is customary for businesses similar to that of the Joint Venture. Each Party shall name the other Party as an additional insured on its commercial general liability policy and shall provide certificates of insurance evidencing such coverage upon request.",
            "Section 11.07. Mitigation. Each Party shall use commercially reasonable efforts to mitigate any damages for which it seeks indemnification or recovery hereunder. No Party shall be entitled to recover damages to the extent that such damages could have been avoided through reasonable mitigation efforts.",
        ],
    ),
    (
        "ARTICLE XII — TERMINATION",
        [
            "Section 12.01. Term. This Agreement shall continue in effect for the term of the Joint Venture as set forth in Section 2.03, unless earlier terminated in accordance with this Article XII.",
            "Section 12.02. Termination Events. The following shall constitute Termination Events: (a) the material breach by a Party of any of its representations, warranties, covenants, or agreements hereunder, which breach (if curable) is not cured within thirty (30) days after written notice thereof; (b) the bankruptcy, insolvency, receivership, or assignment for the benefit of creditors of a Party; (c) the loss by a Party of any license, registration, or qualification material to the performance of its obligations hereunder; (d) the occurrence of a Change of Control of a Party without the prior written consent of the other Party; (e) the occurrence of an event of default under the Senior Credit Facility that is not waived or cured within the applicable grace period; (f) a final, non-appealable judgment against a Party in respect of fraud, willful misconduct, or material breach of fiduciary duty; and (g) such other events as the Parties may from time to time agree in writing to constitute Termination Events.",
            "Section 12.03. Termination Remedies. Upon the occurrence of a Termination Event, the non-defaulting Party shall have the right to (a) terminate this Agreement upon written notice to the defaulting Party; (b) pursue all remedies available at Law or in equity; (c) initiate the buy-sell procedures set forth in Section 12.04; and (d) recover from the defaulting Party all costs and expenses (including reasonable attorneys' fees) incurred in connection with the enforcement of its rights hereunder.",
            "Section 12.04. Buy-Sell Procedure. The non-defaulting Party may, at its option, deliver to the defaulting Party a written notice (a \"Buy-Sell Notice\") specifying a price per unit of Capital Account interest. The defaulting Party shall have thirty (30) days to elect either (a) to purchase the non-defaulting Party's Capital Account interest at such price, or (b) to sell its Capital Account interest to the non-defaulting Party at such price. If the defaulting Party fails to make an election within such thirty (30) day period, it shall be deemed to have elected to sell.",
            "Section 12.05. Liquidation. If the Joint Venture is to be wound up and liquidated upon termination, (a) the Operating Committee shall appoint a liquidator (the \"Liquidator\"); (b) the Liquidator shall promptly proceed to liquidate the assets of the Joint Venture in an orderly manner; (c) the proceeds of liquidation shall be applied in the order of priority set forth in Section 12.06; and (d) the Joint Venture shall be dissolved upon the completion of liquidation and the filing of a certificate of cancellation with the Delaware Secretary of State.",
            "Section 12.06. Order of Payment Upon Liquidation. The proceeds of liquidation shall be applied in the following order: (a) first, to the payment of the costs and expenses of liquidation; (b) second, to the payment of all creditors of the Joint Venture (other than the Parties in their capacity as members) in accordance with their respective priorities; (c) third, to the establishment of reserves for contingent liabilities; (d) fourth, to the payment of any amounts owed to the Parties (other than in their capacity as members), including amounts owed under loans, indemnification obligations, and similar arrangements; and (e) fifth, to the Parties in proportion to their positive Capital Account balances.",
            "Section 12.07. Effect of Termination. Termination of this Agreement shall not relieve any Party from liability for any breach occurring prior to such termination, nor shall it affect any provision hereof that by its nature is intended to survive termination, including without limitation Articles X (Indemnification), XI (Limitation of Liability), XIII (Confidentiality), XIV (Dispute Resolution), and XV (Miscellaneous).",
        ],
    ),
    (
        "ARTICLE XIII — CONFIDENTIALITY",
        [
            "Section 13.01. Confidentiality Obligation. Each Party shall (a) hold all Confidential Information of the other Party in strict confidence; (b) use such Confidential Information solely for the purpose of performing its obligations under this Agreement; (c) not disclose such Confidential Information to any third party without the prior written consent of the disclosing Party, except as expressly permitted hereunder; and (d) implement and maintain reasonable safeguards to protect such Confidential Information from unauthorized access, use, or disclosure.",
            "Section 13.02. Permitted Disclosures. Notwithstanding Section 13.01, a Party may disclose Confidential Information (a) to its directors, officers, employees, agents, and advisors who have a need to know such information and who are bound by confidentiality obligations no less restrictive than those set forth herein; (b) as required by applicable Law, court order, or regulatory authority, provided that the disclosing Party shall (to the extent legally permissible) provide the other Party with prompt written notice and an opportunity to seek a protective order; (c) in connection with the enforcement of its rights hereunder; and (d) to potential transferees of such Party's Capital Account interest, provided that such transferees execute a non-disclosure agreement in form and substance reasonably satisfactory to the other Party.",
            "Section 13.03. Excluded Information. The confidentiality obligations set forth in this Article XIII shall not apply to information that (a) is or becomes generally available to the public through no fault of the receiving Party; (b) was already in the receiving Party's possession on a non-confidential basis prior to disclosure by the disclosing Party; (c) is independently developed by the receiving Party without reference to the disclosing Party's Confidential Information; or (d) is received by the receiving Party from a third party not under an obligation of confidentiality to the disclosing Party.",
            "Section 13.04. Return or Destruction. Upon termination of this Agreement, each Party shall, at the request of the other Party, either return or destroy all Confidential Information of the other Party in its possession or control, and shall certify in writing to the disclosing Party that it has done so; provided, however, that the receiving Party may retain (a) one (1) archival copy of such Confidential Information for legal and regulatory compliance purposes, and (b) Confidential Information contained in routine backup tapes or other electronic storage media that cannot reasonably be deleted, subject in each case to the continuing confidentiality obligations set forth herein.",
            "Section 13.05. Public Announcements. No Party shall issue any press release or make any public statement concerning this Agreement or the transactions contemplated hereby without the prior written consent of the other Party, except as required by applicable Law or the rules of any securities exchange on which such Party's securities are listed.",
        ],
    ),
    (
        "ARTICLE XIV — DISPUTE RESOLUTION",
        [
            "Section 14.01. Good Faith Negotiation. In the event of any dispute, controversy, or claim arising out of or relating to this Agreement (a \"Dispute\"), the Parties shall first attempt in good faith to resolve such Dispute through senior management negotiations for a period of thirty (30) days following written notice of the Dispute from one Party to the other.",
            "Section 14.02. Mediation. If the Dispute is not resolved through negotiation within the thirty (30) day period referenced in Section 14.01, the Parties shall submit the Dispute to non-binding mediation administered by JAMS in Chicago, Illinois, under its Mediation Rules then in effect. The mediation shall be conducted by a single mediator mutually agreed upon by the Parties.",
            "Section 14.03. Arbitration. If the Dispute is not resolved through mediation within sixty (60) days after the appointment of the mediator, the Dispute shall be finally settled by binding arbitration administered by JAMS in Chicago, Illinois, in accordance with its Comprehensive Arbitration Rules and Procedures then in effect. The arbitration shall be conducted by a panel of three (3) arbitrators, each of whom shall be a retired federal judge or magistrate judge with not less than fifteen (15) years of experience in commercial disputes. The arbitration shall be governed by the Federal Arbitration Act, 9 U.S.C. §§ 1 et seq.",
            "Section 14.04. Provisional Remedies. Notwithstanding the foregoing, either Party may seek provisional relief (including without limitation a temporary restraining order, preliminary injunction, or attachment) from a court of competent jurisdiction in connection with any Dispute, without waiving its right to arbitration under Section 14.03.",
            "Section 14.05. Costs and Fees. Each Party shall bear its own costs and attorneys' fees incurred in connection with any Dispute resolution proceeding, except that the prevailing Party in any arbitration shall be entitled to recover from the non-prevailing Party its reasonable attorneys' fees and costs as determined by the arbitrators.",
            "Section 14.06. Confidentiality of Proceedings. The Parties shall maintain the confidentiality of all proceedings hereunder, including without limitation the existence of any Dispute, the substance of any settlement discussions, and the content of any pleadings, evidence, or awards, except to the extent disclosure is required by applicable Law.",
        ],
    ),
    (
        "ARTICLE XV — MISCELLANEOUS",
        [
            "Section 15.01. Governing Law. This Agreement shall be governed by, and construed in accordance with, the laws of the State of Delaware, without regard to its conflicts of laws principles.",
            "Section 15.02. Entire Agreement. This Agreement, together with the schedules, exhibits, and side letters incorporated herein by reference, constitutes the entire agreement between the Parties with respect to the subject matter hereof and supersedes all prior and contemporaneous agreements, understandings, negotiations, and discussions, whether oral or written.",
            "Section 15.03. Amendments. This Agreement may be amended, modified, or supplemented only by a written instrument executed by each Party.",
            "Section 15.04. Waivers. No waiver of any provision of this Agreement shall be effective unless in writing and signed by the Party against whom such waiver is sought to be enforced. No failure or delay by a Party in exercising any right, power, or remedy hereunder shall operate as a waiver thereof, nor shall any single or partial exercise of any such right, power, or remedy preclude any other or further exercise thereof.",
            "Section 15.05. Assignment. No Party may assign or transfer this Agreement or any of its rights or obligations hereunder, by operation of law or otherwise, without the prior written consent of the other Party; provided, however, that either Party may assign this Agreement to a Permitted Transferee upon prior written notice (but without the consent of the other Party).",
            "Section 15.06. Binding Effect. This Agreement shall be binding upon and inure to the benefit of the Parties and their respective successors and permitted assigns.",
            "Section 15.07. No Third-Party Beneficiaries. Except for the Indemnified Parties identified in Article X, this Agreement is intended solely for the benefit of the Parties and shall not confer any rights or remedies on any other Person.",
            "Section 15.08. Severability. If any provision of this Agreement is held by a court of competent jurisdiction to be invalid, illegal, or unenforceable, the validity, legality, and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.",
            "Section 15.09. Counterparts. This Agreement may be executed in any number of counterparts, each of which shall be deemed an original and all of which together shall constitute one and the same instrument. Delivery of an executed counterpart by facsimile or electronic transmission shall be effective as delivery of an original counterpart.",
            "Section 15.10. Notices. All notices, requests, demands, and other communications hereunder shall be in writing and shall be deemed duly given upon (a) personal delivery, (b) the third Business Day after deposit in the United States mail, postage prepaid, certified or registered with return receipt requested, (c) the next Business Day after delivery to a nationally recognized overnight courier service, or (d) confirmed receipt by electronic mail to the addresses set forth on Schedule 15.10.",
            "Section 15.11. Construction. The headings used in this Agreement are for convenience of reference only and shall not affect the interpretation hereof. The words \"include,\" \"includes,\" and \"including\" shall be deemed to be followed by the phrase \"without limitation.\" References to \"$\" or \"dollars\" mean United States dollars.",
            "Section 15.12. Further Assurances. Each Party shall execute and deliver such additional documents and instruments and take such additional actions as the other Party may reasonably request to effectuate the purposes of this Agreement.",
        ],
    ),
]


SCHEDULE_PREAMBLE = """\

────────────────────────────────────────────────────────────────────────────────
SCHEDULES AND EXHIBITS
────────────────────────────────────────────────────────────────────────────────
The following schedules and exhibits are incorporated herein by reference and
form a part of this Agreement as if fully set forth herein. Inclusion is for
discovery completeness and the operative provisions remain those set forth in
the Articles above.
"""


def schedule_block(num: int, title: str) -> str:
    body_lines: list[str] = []
    for i in range(1, 26):
        body_lines.append(
            f"  {num}.{i:02d}  Item {i} of Schedule {num} ({title}). The Parties acknowledge "
            f"that the matters set forth in this paragraph are subject to the financial "
            f"liability limitations of Article XI and, where applicable, the conflict of "
            f"interest disclosure regime of Article IX. The Operating Committee shall "
            f"review this item not less frequently than annually and shall update it as "
            f"circumstances warrant. Cross-references: Article IV §4.07, Article VII §7.07, "
            f"Article IX §9.05, Article XI §11.04, Article XII §12.02."
        )
    body = "\n".join(body_lines)
    return f"\nSCHEDULE {num} — {title}\n{'-' * 60}\n{body}\n"


def build_document() -> str:
    enc = tiktoken.get_encoding("cl100k_base")

    parts: list[str] = [HEADER]
    for title, sections in ARTICLES:
        parts.append(f"\n────────────────────────────────────────────────────────────────────────────────\n{title}\n────────────────────────────────────────────────────────────────────────────────\n")
        parts.extend(s + "\n" for s in sections)

    parts.append(SCHEDULE_PREAMBLE)

    schedule_titles = [
        "Initial Asset Portfolio Inventory",
        "Eligibility Criteria for Qualified Assets",
        "Disclosed Liens and Encumbrances",
        "Personal Trading Restricted-Issuer List",
        "Approved Lender Syndicate",
        "Compliance Officer Procedures",
        "Annual Conflicts Certification Form",
        "Insurance Coverage Specifications",
        "Permitted Investment Guidelines",
        "Form of Notice of Capital Call",
        "Form of Buy-Sell Notice",
        "Schedule of Side Letters",
        "Form of Annual Audit Engagement",
        "Restricted Territory Definition",
        "Permitted Transferee Procedures",
    ]

    n = 1
    while True:
        text = "".join(parts)
        token_count = len(enc.encode(text))
        if token_count >= TARGET_TOKENS:
            break
        title = schedule_titles[(n - 1) % len(schedule_titles)]
        parts.append(schedule_block(n, title))
        n += 1

    final = "".join(parts)
    return final


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    text = build_document()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    n_chars = len(text)
    n_tokens = len(enc.encode(text))
    print(f"wrote {OUT}: {n_chars:,} chars, ~{n_tokens:,} cl100k tokens")


if __name__ == "__main__":
    main()
