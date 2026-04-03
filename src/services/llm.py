import json

import google.generativeai as genai
import httpx
from pydantic import BaseModel, ValidationError

from src.config.settings import get_settings
from src.schemas.enums import PaymentClassification, RejectionReason, SentimentLabel


class LLMNormalizedOutput(BaseModel):
    normalized_text: str
    summary: str
    sentiment: SentimentLabel
    payment_classification: PaymentClassification | None = None
    rejection_reason: RejectionReason | None = None


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def _fallback_structured_output(self, prompt: str) -> str:
        transcript = prompt.split("Transcript:\n", maxsplit=1)[-1].strip() if "Transcript:\n" in prompt else ""
        return json.dumps(
            {
                "normalized_text": transcript,
                "summary": "",
                "sentiment": "NEUTRAL",
                "payment_classification": None,
                "rejection_reason": None,
            }
        )

    def analyze_transcript(self, transcript: str) -> LLMNormalizedOutput | None:
        prompt = self._build_structured_prompt(transcript)
        output = self._call_with_fallback(prompt)
        if not output:
            return None

        parsed = self._extract_json_object(output)
        if parsed is None:
            return None

        try:
            return LLMNormalizedOutput.model_validate(parsed)
        except ValidationError:
            return None

    def _build_structured_prompt(self, transcript: str) -> str:
        return (
            "You are a compliance assistant for multilingual call-center transcripts. "
            "Normalize Hinglish/Tanglish to clean English while preserving intent. "
            "Generate a concise summary in English and do not repeat the transcript verbatim. "
            "Summary must preserve key numbers (including currency amounts), intent (payment, delay, refusal), and named entities (person/company). "
            "Always include the call purpose and context, especially course or institute inquiry details when present. "
            "Return STRICT JSON only with this exact schema and enum values.\n"
            "{\n"
            "  \"normalized_text\": \"string\",\n"
            "  \"summary\": \"string\",\n"
            "  \"sentiment\": \"POSITIVE|NEUTRAL|NEGATIVE\",\n"
            "  \"payment_classification\": \"EMI|FULL_PAYMENT|PARTIAL_PAYMENT|DOWN_PAYMENT|null\",\n"
            "  \"rejection_reason\": \"BUDGET_CONSTRAINTS|HIGH_INTEREST|ALREADY_PAID|NOT_INTERESTED|NONE|null\"\n"
            "}\n"
            "Do not include markdown or extra keys.\n\n"
            f"Transcript:\n{transcript}"
        )

    def _extract_json_object(self, content: str) -> dict | None:
        content = content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _call_with_fallback(self, prompt: str) -> str:
        if self.settings.openrouter_api_key:
            try:
                return self._openrouter_call(prompt)
            except Exception:
                pass

        if self.settings.gemini_api_key:
            try:
                return self._gemini_call(prompt)
            except Exception:
                pass

        if self.settings.allow_mock_llm:
            return json.dumps(
                {
                    "normalized_text": prompt.split("Transcript:\n", maxsplit=1)[-1].strip() if "Transcript:\n" in prompt else "",
                    "summary": "",
                    "sentiment": "NEUTRAL",
                    "payment_classification": None,
                    "rejection_reason": None,
                }
            )

        return self._fallback_structured_output(prompt)

    def _openrouter_call(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.openrouter_model,
            "messages": [
                {"role": "system", "content": "You are a precise compliance assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=60.0) as client:
            response = client.post(f"{self.settings.openrouter_base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def _gemini_call(self, prompt: str) -> str:
        genai.configure(api_key=self.settings.gemini_api_key)
        model = genai.GenerativeModel(self.settings.gemini_model)
        response = model.generate_content(prompt)
        return response.text or ""
