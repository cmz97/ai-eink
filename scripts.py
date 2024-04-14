import time
from llama_cpp import Llama

st = time.time()
llm = Llama(
      model_path="/home/kevin/ai/models/stablelm-2-zephyr-1_6b.q4_k_m.gguf",
      n_ctx=500, # Uncomment to increase the context window
    #   n_threads=3,
)
print(f'load time {time.time() - st}')
st = time.time()
system_message = "Give me a title for this story, return and only return the title."
prompt = """In the week before their departure to Arrakis, when all the final scurrying about had reached a nearly unbearable frenzy, an old crone came to visit the mother of the boy, Paul.
    It was a warm night at Castle Caladan, and the ancient pile of stone that had served the Atreides family as home for twenty-six generations bore that cooled-sweat feeling it acquired before a change in the weather."""
output = llm(
    #   f"""<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\nTitle:""", # Prompt
      f"""<|system|>\n{system_message}<|endoftext|>\n<|user|>\n{prompt}<|endoftext|>\n<|assistant|>\title:""", # Prompt

      max_tokens=50,
      stop=["</s>", "<|im_end|>", "<|endoftext|>", "\n"], # Stop generating just before the model would generate a new question
      echo=False # Echo the prompt back in the output
) # Generate a completion, can also call create_completion
print(f'inference time {time.time() - st}')
print(output)