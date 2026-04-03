from fastapi.testclient import TestClient

from src.config.settings import get_settings
from src.main import app
from src.routes import call_analytics as route_module
from src.schemas.enums import PaymentClassification, RejectionReason, SentimentLabel
from src.schemas.responses import AnalyticsPayload, CallAnalyticsResponse, SOPChecks


def test_api_contract(monkeypatch) -> None:
    async def _mock_run_async(payload):
        return CallAnalyticsResponse(
            status="success",
            language="Tamil",
            callId=payload.call_id or "CALL-TEST-001",
            transcript="Customer will pay partial amount next week.",
            summary="Agent discussed options; customer agreed to partial payment.",
            sop_validation=SOPChecks(
                greeting=True,
                identification=True,
                problemStatement=True,
                solutionOffering=True,
                closing=True,
                complianceScore=1.0,
                adherenceStatus="FOLLOWED",
            ),
            analytics=AnalyticsPayload(
                paymentPreference=PaymentClassification.PARTIAL_PAYMENT,
                rejectionReason=RejectionReason.NONE,
            ),
            sentiment=SentimentLabel.NEUTRAL,
            sop=SOPChecks(
                greeting=True,
                identification=True,
                problemStatement=True,
                solutionOffering=True,
                closing=True,
                complianceScore=1.0,
                adherenceStatus="FOLLOWED",
            ),
            paymentClassification=PaymentClassification.PARTIAL_PAYMENT,
            rejectionReason=RejectionReason.NONE,
            keywords=["partial", "payment", "next", "week"],
            modelInfo={
                "sttProvider": "mock",
                "llmPrimary": "openrouter",
                "llmFallback": "gemini",
            },
        )

    monkeypatch.setattr(route_module.pipeline, "run_async", _mock_run_async)
    monkeypatch.setattr(route_module.store, "upsert_result", lambda *_: None)

    client = TestClient(app)
    settings = get_settings()

    sample_input = {
        "audio_base64": "QVVESU9fQkFTRTY0X1RFU1RfREFUQV9GT1JfQVBJX0NPTlRSQUNUX1RFU1Q=",
        "audio_format": "mp3",
        "language_hint": "hi-en",
        "call_id": "CALL-CONTRACT-001",
    }

    response = client.post(
        "/api/call-analytics",
        json=sample_input,
        headers={"x-api-key": settings.api_key},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["language"] in {"Tamil", "Hindi"}
    assert data["transcript"] != ""
    assert data["summary"] != ""
    assert data["sop_validation"]["adherenceStatus"] in {"FOLLOWED", "NOT_FOLLOWED", "PARTIALLY_FOLLOWED"}
    assert data["analytics"]["paymentPreference"] in {e.value for e in PaymentClassification}
    assert data["analytics"]["rejectionReason"] in {e.value for e in RejectionReason}
    assert len(data["keywords"]) > 0
    assert data["paymentClassification"] in {e.value for e in PaymentClassification}
    assert data["rejectionReason"] in {e.value for e in RejectionReason}
    assert data["sentiment"] in {e.value for e in SentimentLabel}

