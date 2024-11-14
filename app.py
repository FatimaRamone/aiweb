import os
from dotenv import load_dotenv  # Importa la librería para cargar el archivo .env
from flask import Flask, render_template, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Cargar las variables del archivo .env
load_dotenv()

# Obtener la API Key desde las variables de entorno
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configura tu API Key de OpenAI (si no se carga, se podría manejar un error)
if not openai_api_key:
    raise ValueError("La clave de API de OpenAI no está configurada correctamente.")

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
