"""Parsing utilities."""

import httpx


class CsvParser:
    """Simple CSV parser."""

    def __init__(self, delimiter: str = ","):
        self.delimiter = delimiter

    def parse(self, text: str) -> list:
        lines = text.strip().splitlines()
        if not lines:
            return []
        headers = lines[0].split(self.delimiter)
        return [
            dict(zip(headers, row.split(self.delimiter)))
            for row in lines[1:]
        ]

    def to_csv(self, records: list) -> str:
        if not records:
            return ""
        headers = list(records[0].keys())
        rows = [self.delimiter.join(headers)]
        for rec in records:
            rows.append(self.delimiter.join(str(rec.get(h, "")) for h in headers))
        return "\n".join(rows)


class JsonSchemaValidator:
    """Validates dicts against a simple schema."""

    def __init__(self, schema: dict):
        self.schema = schema

    def validate(self, data: dict) -> bool:
        for field, spec in self.schema.items():
            if spec.get("required") and field not in data:
                return False
        return True


async def fetch_remote_schema(url: str) -> dict:
    """Async fetch of a JSON schema."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return resp.json()


def normalize_keys(data: dict) -> dict:
    """Lowercase all keys in a dict."""
    return {k.lower(): v for k, v in data.items()}
