import os
from dotenv import load_dotenv  # Para cargar las variables de entorno desde el archivo .env
from flask import Flask, render_template, request, jsonify
from langchain_community.chat_models import ChatOpenAI  # Cambié la importación según la advertencia
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Cargar las variables del archivo .env
load_dotenv()

# Obtener las API Keys desde las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")

# Configura tu API Key de OpenAI
if not openai_api_key:
    raise ValueError("La clave de API de OpenAI no está configurada correctamente.")

# Inicializar Pinecone
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)

# Crear el índice si no existe (ajustar las dimensiones según el modelo que estés usando)
if 'my-index' not in pc.list_indexes().names():
    pc.create_index(
        name='my-index',  # Nombre en minúsculas y sin caracteres especiales
        dimension=1536,  # Ajusta la dimensión según el modelo de vectores que estés usando
        metric='euclidean',  # O el tipo de métrica que prefieras
        spec=ServerlessSpec(
            cloud='aws',  # O ajusta el proveedor de la nube según tu preferencia
            region='us-east-1'  # Cambié la región a 'us-east-1'
        )
    )

# Crea el modelo de chat usando LangChain
chat = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Definir una plantilla de pregunta
prompt = PromptTemplate(input_variables=["question"], template="Responde a esta pregunta como si fueras un macarra de coruña: {question}")

# Crear una cadena LLM que usará el modelo de OpenAI
chain = LLMChain(prompt=prompt, llm=chat)

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para obtener la respuesta del chatbot
@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    response = chain.run(question)
    return jsonify({'response': response})

# Iniciar el servidor Flask
if __name__ == '__main__':
    app.run(debug=True)
