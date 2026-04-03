from typing import Literal

from pydantic import BaseModel, Field

from src.schemas.enums import AdherenceStatus, PaymentClassification, RejectionReason, SentimentLabel


class SOPChecks(BaseModel):
    greeting: bool
    identification: bool
    problemStatement: bool
    solutionOffering: bool
    closing: bool
    complianceScore: float = Field(..., ge=0.0, le=1.0)
    adherenceStatus: AdherenceStatus


class AnalyticsPayload(BaseModel):
    paymentPreference: PaymentClassification
    rejectionReason: RejectionReason


class CallAnalyticsResponse(BaseModel):
    status: Literal["success"] = "success"
    language: str
    callId: str
    transcript: str = Field(..., min_length=1)
    summary: str
    sop_validation: SOPChecks
    analytics: AnalyticsPayload
    sentiment: SentimentLabel
    sop: SOPChecks
    paymentClassification: PaymentClassification
    rejectionReason: RejectionReason
    keywords: list[str]
    modelInfo: dict[str, str]
