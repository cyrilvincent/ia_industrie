from mistralai import Mistral

api_key = "XcQT0ja9j17oaAoWjeQXsLfoCwcoU4oV"

def chat(system: str, user: str) -> str:
    chat_response = client.chat.complete(
        model="open-mistral-7b",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return chat_response.choices[0].message.content

client = Mistral(api_key=api_key)
old_system = ""
old_user = ""
print("CTRL + C pour arrêter")
while True:
    system = input(f'System (Entrée pour "{old_system}") > ')
    if system == "":
        system = old_system
    user = input(f'> (Entrée pour "{old_user}") > ')
    if user == "":
        user = old_user
    s = chat(system, user)
    print(s)
    old_system = system
    old_user = user




