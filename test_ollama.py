import ollama

MODEL_NAME = "ml_historia-1:latest"

try:
    prompt = "Resumen de la historia en pocas palabras."
    print(f"📩 Enviando prompt: {prompt}")

    response = ollama.generate(model=MODEL_NAME, prompt=prompt)

    print(f"🔍 Respuesta completa de Ollama:\n{response}")

except Exception as e:
    print(f"❌ Error: {str(e)}")
