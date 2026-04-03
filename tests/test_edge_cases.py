from src.schemas.enums import AdherenceStatus, PaymentClassification, RejectionReason
from src.services.analytics import AnalyticsService
from src.services.nlp import NLPService
from src.services.pipeline import CallAnalyticsPipeline
from src.services.sop import SOPService


def test_silent_or_unclear_transcript_defaults() -> None:
    pipeline = CallAnalyticsPipeline()
    summary = pipeline._fallback_summary("\n\n")
    assert summary == "The call contains a short exchange between the caller and customer regarding a prior inquiry."

    stt_conf = pipeline._stt_confidence("Audio unclear, unable to transcribe fully", True)
    assert stt_conf <= 0.25


def test_no_payment_discussion_defaults() -> None:
    analytics = AnalyticsService()
    text = "Hello this is agent. I am checking your issue and will call back tomorrow. Thank you."
    assert analytics.classify_payment(text) == PaymentClassification.EMI
    assert analytics.rejection_reason(text) == RejectionReason.NONE


def test_incomplete_call_not_followed() -> None:
    sop = SOPService()
    result = sop.validate("Hello. Problem noted.")
    assert result.adherenceStatus == AdherenceStatus.NOT_FOLLOWED


def test_noisy_and_slang_keywords_traceable() -> None:
    nlp = NLPService()
    transcript = "hello bro static noise uhh account issue panren now"
    summary = "account issue discussed"
    keywords = nlp.keywords(transcript, summary)
    source = f"{transcript} {summary}".lower()
    assert all(token in source for token in keywords)
