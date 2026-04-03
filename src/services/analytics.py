from src.schemas.enums import PaymentClassification, RejectionReason


class AnalyticsService:
    def classify_payment(self, text: str, llm_suggestion: PaymentClassification | None = None) -> PaymentClassification:
        lowered = text.lower()
        if "emi" in lowered:
            return PaymentClassification.EMI
        if "full payment" in lowered or "settle today" in lowered:
            return PaymentClassification.FULL_PAYMENT
        if "down payment" in lowered:
            return PaymentClassification.DOWN_PAYMENT
        if "partial" in lowered or "part payment" in lowered:
            return PaymentClassification.PARTIAL_PAYMENT
        if llm_suggestion in {
            PaymentClassification.EMI,
            PaymentClassification.FULL_PAYMENT,
            PaymentClassification.PARTIAL_PAYMENT,
            PaymentClassification.DOWN_PAYMENT,
        }:
            return llm_suggestion
        # No explicit payment signal in transcript: use a stable enum-safe default.
        return PaymentClassification.EMI

    def rejection_reason(self, text: str, llm_suggestion: RejectionReason | None = None) -> RejectionReason:
        lowered = text.lower()
        if "budget" in lowered or "cannot afford" in lowered:
            return RejectionReason.BUDGET_CONSTRAINTS
        if "interest" in lowered and ("high" in lowered or "too much" in lowered):
            return RejectionReason.HIGH_INTEREST
        if "already paid" in lowered or "payment done" in lowered:
            return RejectionReason.ALREADY_PAID
        if "not interested" in lowered or "don't want" in lowered:
            return RejectionReason.NOT_INTERESTED
        if llm_suggestion in {
            RejectionReason.BUDGET_CONSTRAINTS,
            RejectionReason.HIGH_INTEREST,
            RejectionReason.ALREADY_PAID,
            RejectionReason.NOT_INTERESTED,
            RejectionReason.NONE,
        }:
            return llm_suggestion
        return RejectionReason.NONE

    def classification_confidence(
        self,
        text: str,
        payment: PaymentClassification,
        rejection: RejectionReason,
    ) -> float:
        lowered = text.lower()
        strong_payment_terms = ["emi", "full payment", "down payment", "part payment", "partial"]
        strong_reject_terms = ["budget", "interest", "already paid", "not interested", "cannot afford"]

        payment_hits = sum(1 for term in strong_payment_terms if term in lowered)
        rejection_hits = sum(1 for term in strong_reject_terms if term in lowered)

        score = 0.55
        if payment != PaymentClassification.EMI:
            score += 0.15
        if rejection != RejectionReason.NONE:
            score += 0.1
        score += min(payment_hits, 2) * 0.08
        score += min(rejection_hits, 2) * 0.06
        return round(min(score, 0.99), 2)
