from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="llama3.1:8b",
    temperature=0.7,
)

print(llm.predict("Hello, Ollama!"))