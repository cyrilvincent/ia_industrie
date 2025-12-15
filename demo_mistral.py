import os
from mistralai import Mistral

api_key = "XcQT0ja9j17oaAoWjeQXsLfoCwcoU4oV"
model = "open-mistral-7b"

client = Mistral(api_key=api_key)

with open("data/openai/quelle_est_bleue.txt") as f:
    text = f.read()

chat_response = client.chat.complete(
    model = model,
    messages = [
        {"role": "system", "content": "De quoi parle ce texte?"},
        {"role": "user", "content": text},
    ]
)

print(chat_response.choices[0].message.content)