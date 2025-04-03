import os
import httpx

from sandbox.fastapi_test.schema import LogRegModel, Item, Foo


BASE_URL = "http://127.0.0.1:8000"


def api_get(url: str):
    if not url:
        raise ValueError(f"Parameter 'url' = {url} is invalid.")

    with httpx.Client() as client:
        response = client.get(url)

    return response


def api_post(url: str, json: (dict | None) = None) -> httpx.Response:
    if not url:
        raise ValueError(f"Parameter 'url' = {url} is invalid.")

    with httpx.Client() as client:
        response = client.post(url, json=json)

    return response


def create_or_update_log_reg_model(name: str) -> httpx.Response:
    url = os.path.join(BASE_URL, "log_reg_models", name)
    response = api_post(url)

    return response


def get_log_reg_model(name: str) -> LogRegModel:
    url = os.path.join(BASE_URL, "log_reg_models", name)
    response = api_get(url)
    model = LogRegModel.from_bytes(response.content)

    return model


def list_log_reg_models(name: str) -> httpx.Response:
    url = os.path.join(BASE_URL, "log_reg_models", "list")
    response = api_get(url)

    return response


if __name__ == "__main__":
    post_response = create_or_update_log_reg_model(name="classifier")
    print(f"response = {post_response}")
