"""Regenerate the sample PDF (data/docs/fee-and-refund-policy.pdf). Needs reportlab."""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent.parent / "data" / "docs" / "fee-and-refund-policy.pdf"

PAGES = [
    ("Course fees for 2026-27", [
        "Sample document for the AskDocs demo. Northstar Academy is a fictional institute.",
        "The full GATE DA course (evening batch) costs Rs. 38,000, including 18% GST.",
        "The weekend batch of the GATE DA course costs Rs. 34,000, including GST.",
        "The six-week crash course costs Rs. 12,500. The test series on its own costs Rs. 3,999.",
        "Course fees include printed study material and the full mock test series. "
        "The test series only option does not include printed notes.",
    ]),
    ("Instalments and scholarships", [
        "The full course fee can be paid in three instalments: 40% at enrolment, 30% within 45 days "
        "and the remaining 30% within 90 days of enrolment.",
        "A late fee of Rs. 500 is charged for every week an instalment is overdue.",
        "The Northstar Scholarship Test is held on the first Sunday of every month. The top 5% of "
        "candidates get a 50% fee waiver and the next 10% get a 25% fee waiver.",
        "Students with a GATE score above 600 in a previous attempt get a 20% fee waiver.",
        "Scholarships and waivers cannot be combined with each other or with any other discount.",
    ]),
    ("Refund policy", [
        "If you withdraw within 7 days of enrolment, you get a full refund minus a processing fee of Rs. 1,000.",
        "If you withdraw between day 8 and day 30 after enrolment, you get a 50% refund of the fees paid.",
        "No refund is given after 30 days, but you can defer your enrolment as described on the next page.",
        "If printed notes have already been shipped, Rs. 2,500 is deducted from any refund.",
        "Crash course and test series fees are non-refundable once the first class or test has started.",
        "Approved refunds are paid to the original payment method within 14 working days.",
    ]),
    ("Transfers and deferrals", [
        "Students may defer their enrolment to the next year's batch once, for a fee of Rs. 2,000, "
        "if they ask within 60 days of enrolment.",
        "A seat cannot be transferred to another person.",
        "Students can switch from classroom to online mode at any time. Any fee difference is "
        "returned as course credit, not as cash.",
    ]),
]


def main() -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(f"Northstar Academy: {title}", styles["Heading1"]))
        story.append(Spacer(1, 8))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 6))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(OUT), pagesize=A4, title="Northstar Academy fee and refund policy").build(story)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
