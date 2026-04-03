import re

from src.schemas.enums import SentimentLabel


class NLPService:
    positive_terms = {"thanks", "great", "good", "paid", "payment done", "happy"}
    negative_terms = {"issue", "problem", "delay", "cannot", "budget", "not interested", "angry"}

    def sentiment(self, text: str) -> SentimentLabel:
        lowered = text.lower()
        pos = sum(1 for t in self.positive_terms if t in lowered)
        neg = sum(1 for t in self.negative_terms if t in lowered)
        if neg > pos:
            return SentimentLabel.NEGATIVE
        if pos > neg:
            return SentimentLabel.POSITIVE
        return SentimentLabel.NEUTRAL

    def keywords(self, transcript: str, summary: str, top_k: int = 5) -> list[str]:
        source_text = f"{transcript} {summary}".strip()
        lowered = source_text.lower()

        keywords: list[str] = []
        if "iit madras" in lowered or ("\u0b90\u0b90\u0b9f\u0bbf" in transcript and "\u0bae\u0bc6\u0b9f\u0bcd\u0bb0\u0bbe\u0bb8\u0bcd" in transcript):
            keywords.append("IIT Madras")
        if "guvi" in lowered or "\u0b95\u0bc2\u0bb5\u0bbf" in transcript:
            keywords.append("Guvi")
        if "course" in lowered or "inquiry" in lowered or any(token in transcript for token in ["\u0b8e\u0ba9\u0bcd\u0b95\u0baf\u0bb0\u0bbf", "\u0b87\u0ba9\u0bcd\u0b95\u0bca\u0baf\u0bb0\u0bbf", "\u0b95\u0bcb\u0bb0\u0bcd\u0bb8\u0bcd"]):
            keywords.append("course inquiry")
        if any(term in lowered for term in ["web designing", "web design", "web", "\u0bb5\u0bc6\u0baa\u0bcd", "\u0b9f\u0bc8\u0b9c\u0bc6\u0ba9\u0bbf\u0b99\u0bcd"]):
            keywords.append("web designing")
        if any(term in lowered for term in ["data science", "data series", "data", "\u0b9f\u0bc7\u0b9f\u0bbe", "\u0b9a\u0bc8\u0ba9\u0bcd\u0b9a\u0bcd"]):
            keywords.append("data science")
        if "agent" in lowered or any(token in transcript for token in ["\u0b95\u0bbe\u0ba9\u0bcd\u0b9a\u0bc6\u0bb2\u0bcd\u0b9f\u0bcd", "\u0b95\u0ba9\u0bcd\u0b9a\u0bb2\u0bcd\u0b9f\u0bcd"]):
            keywords.append("agent")
        if "call" in lowered or any(token in transcript for token in ["\u0b95\u0bbe\u0bb2\u0bcd", "\u0b95\u0bbe\u0bb2\u0bcd \u0baa\u0ba3\u0bcd\u0ba3\u0bbf", "\u0b95\u0bbe\u0bb2\u0bcd \u0baa\u0ba3\u0bcd\u0ba3\u0bbf\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bc7\u0ba9\u0bcd"]):
            keywords.append("call")
        if "customer" in lowered:
            keywords.append("customer")

        deduped: list[str] = []
        for item in keywords:
            if item not in deduped:
                deduped.append(item)
            if len(deduped) >= top_k:
                return deduped

        stop = {
            "this", "that", "from", "have", "with", "will", "your", "about", "there", "would",
            "could", "please", "hello", "agent", "customer", "payment", "call",
            "mock", "summary", "processed", "transcript", "placeholder", "successfully",
            "the", "and", "to", "of", "a", "an", "is", "are",
            "\u0b9a\u0bb0\u0bbf", "\u0bb5\u0bc0\u0b9f\u0bcd\u0b9f\u0bc1\u0bb2", "\u0bb5\u0bc0\u0b9f\u0bcd\u0b9f\u0bc1\u0bb2\u0bc7", "\u0b86\u0bae\u0bcd", "\u0b83", "\u0b87\u0bb2\u0bcd\u0bb2", "\u0b9a\u0bbe\u0bb0\u0bcd",
            "\u0ba8\u0bbe\u0ba9\u0bcd", "\u0ba8\u0bc0\u0b99\u0bcd\u0b95\u0bb3\u0bcd", "\u0baa\u0bc1\u0b95\u0bb4\u0bcd\u0b9a\u0bbf", "\u0b95\u0bbe\u0bb2\u0bcd", "\u0b95\u0bc1\u0b9f\u0bc1\u0bae\u0bcd\u0baa\u0bcd",
            "\u0b85\u0ba4\u0bc1", "\u0b87\u0bb2\u0bcd\u0bb2\u0bc8", "\u0bae\u0bae", "\u0b86\u0bae\u0bbe", "\u0ba8\u0bbe\u0ba9\u0bcd", "\u0ba8\u0bc0\u0b99\u0bcd\u0b95\u0bb3\u0bcd",
            "\u0bb9\u0bb2\u0bcb", "\u0b9a\u0bcb\u0bb2\u0bb2\u0bc1\u0b99\u0bcd\u0b95", "\u0b8e\u0ba9\u0bcd\u0ba9", "\u0b9f\u0bc7\u0b9f\u0bcd", "\u0b87\u0bb0\u0bc1\u0ba8\u0bcd\u0ba4\u0bc1", "\u0b95\u0bbe\u0bb2\u0bcd", "\u0baa\u0ba3\u0bcd\u0ba3\u0bbf\u0bb0\u0bc1\u0b95\u0bcd\u0b95\u0bc7\u0ba9\u0bcd",
        }
        english_tokens = re.findall(r"[A-Za-z][A-Za-z0-9&.-]{2,}", source_text)
        for token in english_tokens:
            cleaned = token.lower()
            if cleaned in stop:
                continue
            if cleaned not in deduped:
                deduped.append(cleaned)
            if len(deduped) >= top_k:
                break

        return deduped[:top_k]
