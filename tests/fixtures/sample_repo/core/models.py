"""Core data models for the sample fixture repo."""

import requests


class User:
    """A user model."""

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def greet(self) -> str:
        return f"Hello, {self.name}!"

    def to_dict(self) -> dict:
        return {"name": self.name, "email": self.email}

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(data["name"], data["email"])

    @staticmethod
    def validate_email(email: str) -> bool:
        return "@" in email


class Product:
    """A product model."""

    def __init__(self, sku: str, price: float):
        self.sku = sku
        self.price = price

    def apply_discount(self, pct: float) -> float:
        return self.price * (1 - pct)

    def fetch_details(self):
        resp = requests.get(f"https://api.example.com/products/{self.sku}")
        return resp.json()


def create_user(name: str, email: str) -> User:
    return User(name, email)


def list_users(base_url: str):
    resp = requests.get(f"{base_url}/users")
    return resp.json()
