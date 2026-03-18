from transformers import pipeline
import time

print("Loading SmolLM-135M-Instruct...")
start = time.time()
pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM-135M-Instruct")
print(f"Loaded in {time.time() - start:.2f}s")

prompt = "Summarize the following professional profile in one short sentence: I am a software engineer with 5 years of experience in Python, AWS, and FastAPI. I have built scalable microservices and led a team of 3 developers."
print("Generating...")
res = pipe(prompt, max_new_tokens=50, return_full_text=False)
print("Result:", res)
