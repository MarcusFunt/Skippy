import pytest
from unittest.mock import patch, MagicMock
from skippy.localai_client import LocalAIClient
import requests

def test_list_models_success():
    client = LocalAIClient("http://localhost:8080")
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"data": [{"id": "model1"}, {"id": "model2"}]}
        mock_get.return_value.raise_for_status = MagicMock()

        models = client.list_models()
        assert models == ["model1", "model2"]

def test_list_models_failure():
    client = LocalAIClient("http://localhost:8080")
    with patch("requests.get") as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        with pytest.raises(RuntimeError) as excinfo:
            client.list_models()
        assert "Failed to list models" in str(excinfo.value)

def test_chat_success():
    client = LocalAIClient("http://localhost:8080")
    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_post.return_value.raise_for_status = MagicMock()

        res = client.chat("gpt-4", [{"role": "user", "content": "hi"}])
        assert res == "Hello!"

def test_chat_failure():
    client = LocalAIClient("http://localhost:8080")
    with patch("requests.post") as mock_post:
        mock_post.side_effect = requests.exceptions.HTTPError("404 Not Found")

        with pytest.raises(RuntimeError) as excinfo:
            client.chat("gpt-4", [{"role": "user", "content": "hi"}])
        assert "LLM chat failed" in str(excinfo.value)
