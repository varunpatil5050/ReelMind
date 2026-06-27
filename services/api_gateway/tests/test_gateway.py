import pytest
from fastapi.testclient import TestClient

from services.api_gateway.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "api_gateway"}

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "reelmind_gateway_requests_total" in response.text
