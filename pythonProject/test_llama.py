from llama_cpp import Llama

llm = Llama(model_path=r"C:\college\Internships\FinGpt\pythonProject\models\mistral-7b-instruct-v0.1.Q4_K_M.gguf")
print(llm("Hello"))
