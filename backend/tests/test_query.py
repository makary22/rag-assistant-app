from fastapi.testclient import TestClient
from app.main import app
from app.services.retrieval import get_retrieval_service

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_query_relevant_question():
    response = client.post(
        "/api/query",
        json={"question": "What is the main topic of these documents?"}
    )

    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "sources" in data
    assert "question" in data
    assert data["question"] == "What is the main topic of these documents?"
    
    # We expect real sources and a real answer to be generated
    assert len(data["answer"]) > 0
    assert len(data["sources"]) > 0

def test_query_irrelevant_question():
    response = client.post(
        "/api/query",
        json={"question": "What is the recipe for making a traditional Italian pizza?"}
    )

    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "sources" in data
    assert data["question"] == "What is the recipe for making a traditional Italian pizza?"
    
    # Depending on the retrieval threshold, sources might be empty for irrelevant questions
    # But the endpoint should still return a valid 200 response and an answer string
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)

def test_query_empty_question():
    response = client.post(
        "/api/query",
        json={"question": ""}
    )
    assert response.status_code in (400, 422)

def test_invalid_input():
    response = client.post(
        "/api/query",
        json={}
    )
    assert response.status_code == 422
