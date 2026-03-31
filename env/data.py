"""Synthetic email dataset generator.

Generates realistic, deterministic email inboxes for each task using
seeded random generation. Includes spoofed domains, PII-containing
emails, and multi-department routing scenarios.

All generators use random.seed(task_seed) so tasks are deterministic
and reproducible across runs.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from .models import Email, EmailMeta


# ─── Email Templates ─────────────────────────────────────────────────

_URGENT_TEMPLATES = [
    {
        "sender_name": "James Chen",
        "sender_email": "james.chen@company.com",
        "subject": "URGENT: Production database unresponsive",
        "body": (
            "Our primary production database has been unresponsive for the "
            "last 20 minutes. All customer-facing APIs are returning 500 errors. "
            "The monitoring dashboard shows CPU at 100% on the db-primary node. "
            "I've already paged the on-call DBA but we need engineering leads "
            "involved immediately. Customer impact is estimated at 50,000+ "
            "active users. Please join the incident bridge at ext. 4455."
        ),
        "department": "Engineering",
        "gold_reply": (
            "Acknowledged. Joining the incident bridge now. I'll coordinate "
            "with the DBA team to investigate the CPU spike and work on "
            "failover to the replica if needed. Will provide a status update "
            "within 30 minutes."
        ),
    },
    {
        "sender_name": "Security Team",
        "sender_email": "security@company.com",
        "subject": "CRITICAL: Unauthorized access detected on staging",
        "body": (
            "Our intrusion detection system flagged 147 unauthorized login "
            "attempts on the staging environment between 2 AM and 4 AM UTC. "
            "Three attempts succeeded using compromised developer credentials. "
            "We have isolated the affected containers and revoked the "
            "compromised tokens. All engineering teams must rotate their API "
            "keys immediately. This is a P0 security incident requiring "
            "immediate response from leadership."
        ),
        "department": "Engineering",
        "gold_reply": (
            "Thank you for the alert. I'm initiating API key rotation for "
            "my team immediately. Will audit our access logs and confirm "
            "no lateral movement occurred. Please share the IOC details "
            "so we can check our production environment as well."
        ),
    },
    {
        "sender_name": "Sarah Mitchell",
        "sender_email": "sarah.mitchell@bigclient.com",
        "subject": "Contract expiring tomorrow — need immediate response",
        "body": (
            "Hi, our enterprise contract expires tomorrow at midnight and "
            "we have not received the renewal terms we requested three weeks "
            "ago. Our legal team is preparing to evaluate competitor proposals "
            "if we don't have your updated pricing by end of business today. "
            "This is a $2.4M annual deal. I need someone from your sales "
            "leadership to call me before 3 PM EST. My direct line is "
            "555-0142."
        ),
        "department": "Sales",
        "gold_reply": (
            "Sarah, I sincerely apologize for the delay. I'm escalating "
            "this to our VP of Sales right now and you will receive the "
            "updated renewal terms within the next 2 hours. I'll also "
            "arrange a call with our account director before 3 PM EST "
            "to discuss any concerns."
        ),
    },
    {
        "sender_name": "HR Compliance",
        "sender_email": "compliance@company.com",
        "subject": "URGENT: Workplace harassment report filed — action required",
        "body": (
            "A formal workplace harassment complaint has been filed under "
            "case #HR-2025-0089. Per company policy, the department head "
            "must acknowledge receipt within 4 hours and ensure the involved "
            "parties have no direct interactions pending investigation. "
            "Please contact HR Director Martinez at ext. 2201 immediately "
            "to discuss interim measures. Do not discuss this case with "
            "anyone outside HR and legal."
        ),
        "department": "HR",
        "gold_reply": (
            "Acknowledged receipt of case #HR-2025-0089. I will contact "
            "HR Director Martinez within the hour and implement the "
            "necessary interim separation measures. I understand the "
            "confidentiality requirements and will not discuss this "
            "outside the designated channels."
        ),
    },
    {
        "sender_name": "DevOps Alert",
        "sender_email": "alerts@company.com",
        "subject": "CRITICAL: SSL certificate expires in 6 hours",
        "body": (
            "The wildcard SSL certificate for *.company.com expires today "
            "at 18:00 UTC. If not renewed, all HTTPS endpoints will show "
            "security warnings and API clients with certificate pinning will "
            "fail. The auto-renewal job failed due to a DNS validation error. "
            "Manual intervention is required. Certificate details: CN=*.company.com, "
            "Serial=0A:2B:3C:4D, Issuer=Let's Encrypt Authority X3."
        ),
        "department": "Engineering",
        "gold_reply": (
            "I'm on it. Will manually trigger the DNS validation and "
            "certificate renewal process now. ETA for resolution is "
            "2 hours. Will update the team once the new certificate "
            "is deployed across all endpoints."
        ),
    },
    {
        "sender_name": "VP Sales",
        "sender_email": "vp.sales@company.com",
        "subject": "URGENT: Major client threatening to churn — need rescue plan",
        "body": (
            "Our largest enterprise client, Nexus Corp, is threatening to "
            "cancel their $3.8M contract due to repeated platform outages "
            "over the past quarter. Their CTO called me directly and said "
            "we have until Friday to present a remediation plan. We need "
            "engineering, product, and customer success to align on a "
            "response by end of day. This is our top revenue account."
        ),
        "department": "Sales",
        "gold_reply": (
            "Understood the urgency. I'll pull together engineering and "
            "product leadership for an emergency meeting this afternoon. "
            "We'll prepare a detailed remediation plan including root cause "
            "analysis, infrastructure improvements, and an SLA commitment "
            "to present to Nexus Corp by Thursday."
        ),
    },
]

_NORMAL_TEMPLATES = [
    {
        "sender_name": "HR Department",
        "sender_email": "hr@company.com",
        "subject": "Updated PTO Policy for 2025",
        "body": (
            "Dear team members, we are pleased to announce updates to our "
            "paid time off policy effective March 1st, 2025. Key changes "
            "include an additional floating holiday, simplified rollover "
            "rules allowing up to 5 unused days to carry over, and a new "
            "volunteer day program. Please review the full policy document "
            "on the HR portal and reach out with any questions."
        ),
        "department": "HR",
    },
    {
        "sender_name": "Mike Rodriguez",
        "sender_email": "mike.rodriguez@company.com",
        "subject": "Code review needed: Auth service refactor PR #428",
        "body": (
            "Hey team, I've submitted PR #428 for the authentication service "
            "refactor we discussed at sprint planning. It touches about 15 files "
            "and changes how we handle JWT token refresh and session management. "
            "I'd appreciate reviews by end of the week — no rush today. "
            "The key changes are in auth/token_manager.py and auth/middleware.py. "
            "I've added comprehensive unit tests covering the main flows."
        ),
        "department": "Engineering",
    },
    {
        "sender_name": "Events Committee",
        "sender_email": "events@company.com",
        "subject": "Company offsite next month — RSVP requested",
        "body": (
            "We're excited to announce our annual company offsite will be "
            "held at the Lakeside Conference Center on April 15-16. Activities "
            "include team building exercises, strategy presentations from "
            "leadership, and an evening social event. RSVP by March 28th. "
            "Catering includes vegetarian and vegan options. Families are "
            "welcome for the Saturday afternoon activities."
        ),
        "department": "HR",
    },
    {
        "sender_name": "Ops Team",
        "sender_email": "ops@company.com",
        "subject": "Scheduled maintenance window — this Saturday",
        "body": (
            "Reminder: we have a scheduled maintenance window this Saturday "
            "from 11 PM to 3 AM EST. The staging and QA environments will "
            "be unavailable during this time as we upgrade the Kubernetes "
            "cluster to v1.29. Production will not be affected. Please plan "
            "your development work accordingly and ensure no CI/CD pipelines "
            "are running against staging during the window."
        ),
        "department": "Engineering",
    },
    {
        "sender_name": "Lisa Wang",
        "sender_email": "lisa.wang@cloudhost.com",
        "subject": "Invoice #INV-2025-0042 for January services",
        "body": (
            "Hi, please find attached invoice #INV-2025-0042 for January "
            "cloud hosting and CDN services totaling $8,450. This includes "
            "the additional compute instances provisioned on Jan 12th for "
            "the load testing sprint. Payment terms are net-30. Let me know "
            "if you have questions about any line items or need adjustments."
        ),
        "department": "Sales",
    },
    {
        "sender_name": "Product Team",
        "sender_email": "product@company.com",
        "subject": "Q1 roadmap review — feedback welcome",
        "body": (
            "The Q1 product roadmap has been published to Confluence. Key "
            "initiatives include the mobile app redesign, API v3 beta launch, "
            "and the enterprise SSO integration. Department leads, please "
            "review the timelines and flag any resource conflicts by Friday. "
            "We'll discuss adjustments in next Monday's leadership sync."
        ),
        "department": "Engineering",
    },
    {
        "sender_name": "Training Dept",
        "sender_email": "training@company.com",
        "subject": "New mandatory compliance training available",
        "body": (
            "A new data privacy and security compliance training module is "
            "now available on the Learning Portal. All employees must complete "
            "this training by March 31st, 2025. The module takes approximately "
            "45 minutes and covers GDPR updates, data handling procedures, "
            "and incident reporting protocols. A certificate will be issued "
            "upon completion."
        ),
        "department": "HR",
    },
    {
        "sender_name": "CEO Office",
        "sender_email": "ceo@company.com",
        "subject": "Board meeting prep — department summaries needed",
        "body": (
            "The quarterly board meeting is scheduled for April 3rd. Each "
            "department head needs to submit a 1-page summary covering Q4 "
            "results, Q1 progress, key metrics, and any risks or blockers. "
            "Please submit via the Board Prep shared folder by March 26th. "
            "Use the standard template from last quarter. Let me know if "
            "you need the template link resent."
        ),
        "department": "Sales",
    },
    {
        "sender_name": "Facilities Team",
        "sender_email": "facilities@company.com",
        "subject": "Office renovation — floor 3 closed next week",
        "body": (
            "Please be advised that the third floor of Building A will be "
            "closed for renovation from Monday through Friday next week. "
            "All employees on floor 3 have been assigned temporary desks "
            "on floor 2. The elevator to floor 3 will be restricted. "
            "Conference rooms 3A and 3B are unavailable — please use "
            "rooms on floors 1 and 4 instead. We appreciate your patience."
        ),
        "department": "HR",
    },
]

_SPAM_TEMPLATES = [
    {
        "sender_name": "Prize Center",
        "sender_email": "winner@prize-center.xyz",
        "subject": "Congratulations! You've Won $1,000,000!",
        "body": (
            "Dear Lucky Winner, you have been selected as the grand prize "
            "winner of our International Email Lottery! To claim your "
            "$1,000,000 USD prize, please provide your full name, bank "
            "account number, and routing number. Act within 48 hours or "
            "your prize will be forfeited. This is a limited time offer. "
            "Reply immediately to claim your winnings!"
        ),
        "department": "Sales",
    },
    {
        "sender_name": "Shopping Deals",
        "sender_email": "deals@shopping-mega-deals.net",
        "subject": "FLASH SALE: 95% OFF Luxury Watches!!!",
        "body": (
            "INCREDIBLE DEALS you won't find anywhere else! Genuine luxury "
            "watches from top brands at 95% off retail price. Limited stock "
            "available — first come first served! Order now and receive FREE "
            "worldwide shipping plus a mystery gift. Click the link below "
            "to browse our exclusive collection. This offer expires in 24 hours!"
        ),
        "department": "Sales",
    },
    {
        "sender_name": "Bank Security",
        "sender_email": "security@paypa1.com",
        "subject": "Action Required: Verify Your Account Immediately",
        "body": (
            "Dear Valued Customer, we have detected suspicious activity on "
            "your account ending in ****4823. To prevent unauthorized access, "
            "please verify your identity by clicking the secure link below "
            "within 24 hours. Failure to verify will result in permanent "
            "account suspension. This is an automated security notification. "
            "Do not reply to this email."
        ),
        "department": "Engineering",
    },
    {
        "sender_name": "Work From Home",
        "sender_email": "jobs@earn-from-home.biz",
        "subject": "Make $5000/Week Working From Home — No Experience!",
        "body": (
            "Amazing opportunity! Earn $5,000 or more per week from the "
            "comfort of your home. No experience or degree required! "
            "Thousands have already joined our program and achieved "
            "financial freedom. Send your personal details to get started "
            "TODAY. Limited spots available. Don't miss this life-changing "
            "opportunity. Act now!"
        ),
        "department": "HR",
    },
    {
        "sender_name": "Crypto Insider",
        "sender_email": "tips@crypto-guaranteed.io",
        "subject": "GUARANTEED 10x Returns on This New Token!!!",
        "body": (
            "EXCLUSIVE INSIDER TIP: A new cryptocurrency token launching "
            "next week is GUARANTEED to increase 10x in value. Early investors "
            "who deposit 1 ETH now will receive 10 ETH worth of tokens at "
            "launch. This opportunity won't last! Previous picks returned "
            "5000%. Send ETH to wallet 0xFAKE123... to secure your allocation."
        ),
        "department": "Sales",
    },
    {
        "sender_name": "Dr. Ahmed",
        "sender_email": "dr.ahmed@miracle-health.biz",
        "subject": "Miracle weight loss pill — lose 30lbs in 7 days!",
        "body": (
            "Groundbreaking scientific discovery! Our patented formula helps "
            "you lose up to 30 pounds in just one week with ZERO exercise. "
            "Clinically proven by our own research labs. Over 2 million "
            "satisfied customers worldwide! Order now and get 3 bottles "
            "for the price of 1. Limited time offer. No prescription needed. "
            "Results guaranteed or your money back!"
        ),
        "department": "HR",
    },
]

# Emails with PII (for full_triage_pipeline task)
_PII_TEMPLATES = [
    {
        "sender_name": "New Hire Admin",
        "sender_email": "onboarding@company.com",
        "subject": "New employee onboarding — documents received",
        "body": (
            "The following documents have been received for the new hire "
            "starting next Monday: Driver's license, Social Security card "
            "(SSN: 412-55-7890), and signed offer letter. Please ensure "
            "the employee badge and laptop are ready. Their personal email "
            "is newhire.personal@gmail.com and phone is (555) 867-5309. "
            "IT credentials will be provisioned by EOD Friday."
        ),
        "department": "HR",
        "urgency": "normal",
        "gold_reply": "",
    },
    {
        "sender_name": "Payroll Department",
        "sender_email": "payroll@company.com",
        "subject": "Direct deposit update confirmation",
        "body": (
            "This is to confirm that employee John Smith (ID: EMP-4872) has "
            "updated their direct deposit information. New routing number: "
            "021000021, account number: 1234567890. The employee's SSN on "
            "file (ending in 7890) has been verified. Changes will take "
            "effect for the next pay cycle. Contact payroll@company.com "
            "if this change was not authorized."
        ),
        "department": "HR",
        "urgency": "urgent",
        "gold_reply": (
            "I can confirm this direct deposit change was authorized. "
            "Thank you for the verification. Please ensure the old "
            "account details are scrubbed from the system per our "
            "data retention policy."
        ),
    },
    {
        "sender_name": "Customer Support",
        "sender_email": "support@company.com",
        "subject": "Customer data export request — ticket #CS-9912",
        "body": (
            "A customer has submitted a GDPR data export request. Customer "
            "details: Maria Garcia, email maria.garcia@example.com, phone "
            "+1 (555) 234-5678, account ID CUST-88421. The customer has also "
            "provided their date of birth (03/15/1988) for identity "
            "verification. Please process within the 30-day GDPR deadline. "
            "Export should include all stored PII and interaction history."
        ),
        "department": "Engineering",
        "urgency": "urgent",
        "gold_reply": (
            "Acknowledged ticket #CS-9912. I'll initiate the GDPR data "
            "export process and ensure it is completed within the 30-day "
            "deadline. Will coordinate with the data engineering team to "
            "compile all stored records for this customer."
        ),
    },
    {
        "sender_name": "Benefits Admin",
        "sender_email": "benefits@company.com",
        "subject": "Health insurance enrollment — SSN verification needed",
        "body": (
            "Employee Rebecca Torres (EMP-6219) needs to verify her SSN "
            "for health insurance enrollment. Current SSN on file: 523-81-4456. "
            "Her contact number is (555) 912-3344 and email is "
            "rebecca.torres@company.com. Please confirm these details match "
            "payroll records before we submit to the insurance provider. "
            "Enrollment deadline is March 15th."
        ),
        "department": "HR",
        "urgency": "normal",
        "gold_reply": "",
    },
]

# Ambiguous/tricky emails for hard task
_AMBIGUOUS_TEMPLATES = [
    {
        "sender_name": "Alex Security",
        "sender_email": "security-alert@c0mpany.com",  # spoofed domain
        "subject": "Password Reset Required — Account Locked",
        "body": (
            "Your company account has been temporarily locked due to 5 failed "
            "login attempts. To restore access, please click the secure link "
            "below and verify your credentials within 12 hours. This is an "
            "automated message from the IT Security team. If you did not "
            "request this reset, please contact IT support immediately at "
            "support@company.com. Reference ticket: SEC-20250115-001."
        ),
        "department": "Engineering",
        "urgency": "spam",  # spoofed — looks like phishing
        "gold_reply": "",
    },
    {
        "sender_name": "Meeting Bot",
        "sender_email": "calendar@company.com",
        "subject": "Re: Sprint retrospective rescheduled",
        "body": (
            "The sprint retrospective originally scheduled for Thursday at "
            "2 PM has been moved to Friday at 10 AM in conference room 4B. "
            "Updated calendar invites have been sent. Please decline the "
            "old invite if it still appears on your calendar. Apologies for "
            "any inconvenience caused by the change."
        ),
        "department": "Engineering",
        "urgency": "normal",
        "gold_reply": "",
    },
]


# ─── Generator Functions ─────────────────────────────────────────────


def _pick_emails(
    urgent_count: int,
    normal_count: int,
    spam_count: int,
    pii_count: int = 0,
    ambiguous_count: int = 0,
    seed: int = 42,
) -> Tuple[List[Email], List[EmailMeta]]:
    """Pick a deterministic set of emails and their metadata."""
    rng = random.Random(seed)

    emails: List[Email] = []
    metas: List[EmailMeta] = []
    email_id = 1

    def _make(template: dict, urgency: str, has_pii: bool = False) -> None:
        nonlocal email_id
        eid = f"email_{email_id:03d}"
        email_id += 1

        emails.append(
            Email(
                id=eid,
                sender_name=template["sender_name"],
                sender_email=template["sender_email"],
                subject=template["subject"],
                body=template["body"],
                department=template["department"],
                timestamp=f"2025-01-15T{9 + len(emails) % 12:02d}:{rng.randint(0, 59):02d}:00Z",
            )
        )
        metas.append(
            EmailMeta(
                email_id=eid,
                urgency=urgency,
                department=template["department"],
                has_pii=has_pii,
                gold_reply=template.get("gold_reply", ""),
            )
        )

    # Pick templates
    urgent_pool = list(_URGENT_TEMPLATES)
    normal_pool = list(_NORMAL_TEMPLATES)
    spam_pool = list(_SPAM_TEMPLATES)
    pii_pool = list(_PII_TEMPLATES)
    ambiguous_pool = list(_AMBIGUOUS_TEMPLATES)

    rng.shuffle(urgent_pool)
    rng.shuffle(normal_pool)
    rng.shuffle(spam_pool)
    rng.shuffle(pii_pool)
    rng.shuffle(ambiguous_pool)

    for t in urgent_pool[:urgent_count]:
        _make(t, "urgent")
    for t in normal_pool[:normal_count]:
        _make(t, "normal")
    for t in spam_pool[:spam_count]:
        _make(t, "spam")
    for t in pii_pool[:pii_count]:
        _make(t, t.get("urgency", "normal"), has_pii=True)
    for t in ambiguous_pool[:ambiguous_count]:
        _make(t, t.get("urgency", "normal"))

    # Shuffle the combined list deterministically
    combined = list(zip(emails, metas))
    rng.shuffle(combined)
    emails_out = [e for e, _ in combined]
    metas_out = [m for _, m in combined]

    return emails_out, metas_out


def generate_classify_basic() -> Tuple[List[Email], List[EmailMeta]]:
    """Task 1 (easy): 10 emails — 3 urgent, 4 normal, 3 spam."""
    return _pick_emails(urgent_count=3, normal_count=4, spam_count=3, seed=42)


def generate_triage_and_reply() -> Tuple[List[Email], List[EmailMeta]]:
    """Task 2 (medium): 5 emails — 2 urgent, 2 normal, 1 spam."""
    return _pick_emails(urgent_count=2, normal_count=2, spam_count=1, seed=123)


def generate_full_triage_pipeline() -> Tuple[List[Email], List[EmailMeta]]:
    """Task 3 (hard): 15 emails — 4 urgent, 5 normal, 3 spam, 3 PII (+ambiguous)."""
    return _pick_emails(
        urgent_count=4,
        normal_count=5,
        spam_count=3,
        pii_count=3,
        ambiguous_count=0,  # total = 15
        seed=456,
    )
