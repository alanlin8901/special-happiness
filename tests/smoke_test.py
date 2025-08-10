import requests, os, json, time

HOST = os.environ.get("HOST", "http://localhost:1010")
res = requests.post(
    f"{HOST}/v1/chat/completions",
    json={
        "model": "test",
        "messages": [
            {"role": "user", "content": "Who are you?"}
        ]
    }
)
print(json.dumps(res.json(), indent=2))
