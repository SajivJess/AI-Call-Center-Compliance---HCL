import json
import sqlite3
from pathlib import Path

from src.config.settings import get_settings
from src.schemas.responses import CallAnalyticsResponse


class SQLiteStore:
    def __init__(self) -> None:
        settings = get_settings()
        db_path = Path(settings.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS call_analytics (
                    call_id TEXT PRIMARY KEY,
                    transcript TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def upsert_result(self, result: CallAnalyticsResponse) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO call_analytics(call_id, transcript, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(call_id) DO UPDATE SET
                    transcript = excluded.transcript,
                    payload_json = excluded.payload_json,
                    created_at = CURRENT_TIMESTAMP
                """,
                (
                    result.callId,
                    result.transcript,
                    json.dumps(result.model_dump(), ensure_ascii=True),
                ),
            )
            conn.commit()
