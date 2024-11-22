from openai import OpenAI

with open("data/openai/openai.env") as f:
    key = f.read()

with open("openai_code.py") as f:
    text = f.read()

#print(text)
# copilot

client = OpenAI(api_key=key)

completion = client.chat.completions.create(model="gpt-3.5-turbo", max_tokens=1000,
    messages=[
        {"role": "system", "content": "Explique moi ca que fais ce code Python"},
        {"role": "user", "content": text}
    ]
)

res = completion.choices[0].message
print(res.content)
