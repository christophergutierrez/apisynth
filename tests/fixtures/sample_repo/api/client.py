"""API client module."""

import requests


class ApiClient:
    """Generic REST API client."""

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def get(self, path: str, **kwargs):
        return self.session.get(f"{self.base_url}{path}", **kwargs)

    def post(self, path: str, data=None, **kwargs):
        return self.session.post(f"{self.base_url}{path}", json=data, **kwargs)

    def put(self, path: str, data=None, **kwargs):
        return self.session.put(f"{self.base_url}{path}", json=data, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.session.delete(f"{self.base_url}{path}", **kwargs)

    def paginate(self, path: str, page_size: int = 100):
        page = 0
        while True:
            resp = self.get(path, params={"page": page, "size": page_size})
            data = resp.json()
            if not data:
                break
            yield data
            page += 1


def fetch_weather(city: str) -> dict:
    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city},
    )
    return resp.json()


def post_event(url: str, payload: dict) -> bool:
    resp = requests.post(url, json=payload)
    return resp.status_code == 200
