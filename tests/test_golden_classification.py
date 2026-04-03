from src.schemas.enums import AdherenceStatus, PaymentClassification, RejectionReason
from src.services.analytics import AnalyticsService
from src.services.nlp import NLPService
from src.services.sop import SOPService


def _assert_keywords_traceable(keywords: list[str], transcript: str, summary: str) -> None:
    source = f"{transcript} {summary}".lower()
    for token in keywords:
        assert token in source


def test_golden_hinglish_samples() -> None:
    sop = SOPService()
    nlp = NLPService()
    analytics = AnalyticsService()

    samples = [
        {
            "text": "Namaste this is agent Ravi. I am calling about your EMI issue. We can resolve by enabling EMI payment. Thank you bye.",
            "summary": "Agent discussed EMI resolution and closed politely.",
            "payment": PaymentClassification.EMI,
            "reason": RejectionReason.NONE,
        },
        {
            "text": "Hello this is agent Neha regarding your account problem. I will share full payment option. Thank you have a nice day.",
            "summary": "Agent offered full payment and closed call.",
            "payment": PaymentClassification.FULL_PAYMENT,
            "reason": RejectionReason.NONE,
        },
        {
            "text": "Hi I am your agent. Problem is pending amount. We can do part payment this month. Thank you bye.",
            "summary": "Part payment plan offered.",
            "payment": PaymentClassification.PARTIAL_PAYMENT,
            "reason": RejectionReason.NONE,
        },
        {
            "text": "Namaste this is agent Sita. About your account issue, we can start with down payment. Thank you.",
            "summary": "Down payment discussed.",
            "payment": PaymentClassification.DOWN_PAYMENT,
            "reason": RejectionReason.NONE,
        },
        {
            "text": "Hello this is agent Karan regarding your account problem. Customer says budget is low and not interested. Thank you bye.",
            "summary": "Customer declined due to budget.",
            "payment": PaymentClassification.EMI,
            "reason": RejectionReason.BUDGET_CONSTRAINTS,
        },
    ]

    for sample in samples:
        sop_result = sop.validate(sample["text"])
        assert sop_result.complianceScore == (
            sum([
                sop_result.greeting,
                sop_result.identification,
                sop_result.problemStatement,
                sop_result.solutionOffering,
                sop_result.closing,
            ])
            / 5
        )
        assert sop_result.adherenceStatus in {AdherenceStatus.FOLLOWED, AdherenceStatus.NOT_FOLLOWED}

        payment = analytics.classify_payment(sample["text"])
        reason = analytics.rejection_reason(sample["text"])
        assert payment == sample["payment"]
        assert reason == sample["reason"]

        keywords = nlp.keywords(sample["text"], sample["summary"])
        _assert_keywords_traceable(keywords, sample["text"], sample["summary"])


def test_golden_tanglish_samples() -> None:
    sop = SOPService()
    nlp = NLPService()
    analytics = AnalyticsService()

    samples = [
        {
            "text": "Vanakkam this is agent Arun. Account issue pathi pesuren. We can do EMI option. Thank you good day.",
            "summary": "EMI option provided.",
            "payment": PaymentClassification.EMI,
            "reason": RejectionReason.NONE,
        },
        {
            "text": "Hello this is agent Priya regarding your account problem. Full payment pannina close agum. Thank you bye.",
            "summary": "Full payment recommendation.",
            "payment": PaymentClassification.FULL_PAYMENT,
            "reason": RejectionReason.NONE,
        },
        {
            "text": "Hi I am your agent. Pending issue iruku. Part payment option kudukalam. Thank you.",
            "summary": "Part payment option.",
            "payment": PaymentClassification.PARTIAL_PAYMENT,
            "reason": RejectionReason.NONE,
        },
        {
            "text": "Vanakkam my name is Bala. Problem regarding your account. Start with down payment then continue. Thank you bye.",
            "summary": "Down payment first strategy.",
            "payment": PaymentClassification.DOWN_PAYMENT,
            "reason": RejectionReason.NONE,
        },
        {
            "text": "Hello this is agent Devi regarding your account issue. Customer says high interest and not interested. Thank you.",
            "summary": "Customer rejected due to high interest.",
            "payment": PaymentClassification.EMI,
            "reason": RejectionReason.HIGH_INTEREST,
        },
    ]

    for sample in samples:
        sop_result = sop.validate(sample["text"])
        assert sop_result.adherenceStatus in {AdherenceStatus.FOLLOWED, AdherenceStatus.NOT_FOLLOWED}

        payment = analytics.classify_payment(sample["text"])
        reason = analytics.rejection_reason(sample["text"])
        assert payment == sample["payment"]
        assert reason == sample["reason"]

        keywords = nlp.keywords(sample["text"], sample["summary"])
        _assert_keywords_traceable(keywords, sample["text"], sample["summary"])
