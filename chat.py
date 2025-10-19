from openai import OpenAI

def get_api_key():
  with open('openai_key.txt', 'r') as f:
    return f.read().strip()

client = OpenAI(api_key=get_api_key())
messages = []

while True:
  user_prompt = input("Prompt: ")
  messages.append({"role": "user", "content": user_prompt})

  response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages
  )
  assistant_response = response.choices[0].message.content
  messages.append({"role": "assistant", "content": assistant_response})
  print(assistant_response)
  print("\n",messages,"\n")