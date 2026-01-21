import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

resp = client.responses.create(
    model= "gpt-4.1-2025-04-14",
    input= "Tell me something interesting about AI bills in the US"
)

print(resp.output_text)
#Gemini