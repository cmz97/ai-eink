
import json
import random
import requests
import time

st = time.time()
prompt =  """<|system|>
give me a title for this story<|endoftext|>
<|user|>
One day, a boy named Tom was playing in the garden. He had a special stone, he liked to keep it in his pocket. Suddenly, he felt something tickling his face. It was a mosquito! He tried to swat it away, but it kept coming back.
Tom's dad saw the mosquito and said, "Tom, why don't you give the mosquito a gift?" Tom thought for a minute, then he remembered the stone he had been keeping in his pocket. He took it out and held it up to the mosquito.
The mosquito flew onto the stone and stayed there. Tom smiled and said, "I'm so glad I kept the stone. Now the mosquito has a place to stay!"
Tom's dad smiled too. He said, "You're so kind, Tom. You showed that even the smallest things can make a big difference."
Tom nodded and hugged his dad. He was glad he had been able to help the mosquito.
<|assistant|>
title: 
"""
data = {
    "prompt": prompt,
    "model": "stablelm2",
    "stream": False,
    "options": {
        "temperature": 1.5, 
        "top_p": 0.99, 
        "top_k": 100,
        "num_ctx" : 500,
        "num_predict": 25,
    },
}
response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
json_data = json.loads(response.text)["response"]
text = json_data.replace("\n", ",").replace('"', '').replace(".", ",")
print(f" --------- {time.time() - st} --------\n\n")
print(text)