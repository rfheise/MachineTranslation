from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='openai-gpt')
set_seed(42)

text = ""
while True:
    print("Enter Text: ",end="")
    text += input()
    out = generator(text, max_new_tokens=100, truncation=True, num_return_sequences=1)[0]['generated_text']
    print(f"\n\nGPT: {out.strip(text)}\n")
    text += out