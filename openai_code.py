from openai import OpenAI

with open("data/openai/openai.env") as f:
    key = f.read()

with open("openai_code.py") as f:
    text = f.read()

#print(text)
# copilot

client = OpenAI(api_key=key)

nb_points = 3

completion = client.chat.completions.create(model="gpt-4o", max_tokens=1000,
    messages=[
        {"role": "system", "content": f"Explique moi ca que fais ce code Python en {nb_points} points et 50 mots max"},
        {"role": "user", "content": text}
    ]
)

res = completion.choices[0].message
print(res.content)
