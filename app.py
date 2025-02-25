from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import ollama

app = Flask(__name__)
CORS(app)  # Habilita CORS para evitar problemas de comunicación con el frontend

# Página principal (carga index.html)
@app.route("/")
def index():
    return render_template("index.html")

# Ruta para recibir preguntas desde el frontend
@app.route("/api/ask", methods=["POST"])
def ask_model():
    try:
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({"response": "Por favor, ingresa una pregunta."}), 400

        # Llamada a Ollama
        response = ollama.chat(model="ml_historia-1:latest", messages=[{"role": "user", "content": question}])

        # Extraer la respuesta del modelo
        answer = response.get("message", "No se obtuvo respuesta.")

        return jsonify({"response": answer})
    
    except Exception as e:
        return jsonify({"response": f"Error al procesar la pregunta: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
