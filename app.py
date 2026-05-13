from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3:4b', messages=[
  {
    'role': 'user',
    'content': 'what is the differnce between generate and chat',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)
