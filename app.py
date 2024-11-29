import os
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# Ignorar advertencias innecesarias
warnings.filterwarnings("ignore", category=UserWarning, module='langchain')

# Cargar variables de entorno
load_dotenv()

# Configuración de claves de OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("La clave API de OpenAI no está configurada.")

# Crear la aplicación Flask
app = Flask(__name__, static_folder="dist", static_url_path="")
CORS(app)  # Permitir CORS para frontend-backend

# Configurar LangChain con OpenAI
chat = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(
    input_variables=["question", "documents"],
    template="Responde a esta pregunta usando la información de los siguientes documentos: {documents}. Pregunta: {question}"
)
chain = LLMChain(prompt=prompt, llm=chat)

# Leer documentos desde "assets/documento.txt"
def load_documents(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.readlines()
    except FileNotFoundError:
        raise ValueError(f"No se encontró el archivo {filepath}. Asegúrate de que exista y esté en la ubicación correcta.")

documents_path = os.path.join("dist", "assets", "documento.txt")
documents = load_documents(documents_path)

# Fragmentar texto para evitar exceder límites de tokens
def chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

chunked_documents = []
for doc in documents:
    chunked_documents.extend(chunk_text(doc))

# Encontrar documentos relevantes
def find_relevant_docs(question, documents):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    ranked_indices = similarities.argsort()[::-1][:3]
    return [documents[i] for i in ranked_indices]

# Rutas para servir el frontend
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

# Ruta para el chatbot
@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question")
    if not question:
        return jsonify({"error": "No se proporcionó una pregunta."}), 400

    # Buscar documentos relevantes
    relevant_docs = find_relevant_docs(question, chunked_documents)
    response = chain.run(question=question, documents=" ".join(relevant_docs))
    return jsonify({"response": response})

# Iniciar el servidor
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
