from enum import Enum


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class PaymentClassification(str, Enum):
    EMI = "EMI"
    FULL_PAYMENT = "FULL_PAYMENT"
    PARTIAL_PAYMENT = "PARTIAL_PAYMENT"
    DOWN_PAYMENT = "DOWN_PAYMENT"


class RejectionReason(str, Enum):
    BUDGET_CONSTRAINTS = "BUDGET_CONSTRAINTS"
    HIGH_INTEREST = "HIGH_INTEREST"
    ALREADY_PAID = "ALREADY_PAID"
    NOT_INTERESTED = "NOT_INTERESTED"
    NONE = "NONE"


class AdherenceStatus(str, Enum):
    FOLLOWED = "FOLLOWED"
    PARTIALLY_FOLLOWED = "PARTIALLY_FOLLOWED"
    NOT_FOLLOWED = "NOT_FOLLOWED"
