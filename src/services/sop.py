from src.schemas.enums import AdherenceStatus
from src.schemas.responses import SOPChecks


class SOPService:
    rules = {
        "greeting": ["hello", "hi", "namaste", "vanakkam", "ஹலோ", "வணக்கம்"],
        "identification": ["this is", "my name", "agent", "i am", "நான்", "என் பெயர்", "நா"],
        "problemStatement": ["issue", "problem", "regarding", "about your account"],
        "solutionOffering": ["we can", "i will", "solution", "option", "resolve"],
        "closing": ["thank you", "have a nice day", "bye", "good day", "நன்றி", "பிரியாவிடை"],
    }

    def validate(self, text: str) -> SOPChecks:
        lowered = text.lower()
        checks: dict[str, bool] = {}
        for step, keywords in self.rules.items():
            checks[step] = any(k in lowered for k in keywords)

        score = sum([checks["greeting"], checks["identification"], checks["problemStatement"], checks["solutionOffering"], checks["closing"]]) / 5
        status = AdherenceStatus.FOLLOWED if all(checks.values()) else AdherenceStatus.NOT_FOLLOWED

        return SOPChecks(
            greeting=checks["greeting"],
            identification=checks["identification"],
            problemStatement=checks["problemStatement"],
            solutionOffering=checks["solutionOffering"],
            closing=checks["closing"],
            complianceScore=score,
            adherenceStatus=status,
        )
