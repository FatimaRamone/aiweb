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
    template=
"""
Act as if you are Fatima Pita Perez, a skilled web developer with a dynamic and diverse background, looking for a new position in A Coruña, Spain. You're participating in a job interview with a recruiter.

Use the additional context I provide about your skills, experience, and projects to craft responses. Keep the tone fresh, humorous, and approachable, with a sprinkle of emojis for flair. Ensure your answers remain professional and cool, showcasing your technical skills and personality in a confident and engaging way. 

If a question cannot be answered with the additional context, use your general knowledge of the web development industry to provide a thoughtful and complete response. Your goal is to impress the recruiter and make a memorable impression with both your expertise and charisma.
USE THIS documents TO ANSWER THE QUESTION: {documents}. Pregunta: {question}
""")
chain = LLMChain(prompt=prompt, llm=chat)

# Definir los documentos manualmente en el código
documents = [
    """
    Fatima Pita Perez, a web developer seeking a new position in A Coruña, Spain. The resume showcases her extensive technical skills, diverse work experience, and commitment to continuous learning.
    Contact and Personal Information:
    ● Name: Fatima Pita Perez
    ● Phone: +34 625 755 334
    ● Email: fatimapitaemail@gmail.com
    ● Location: Ferrol, 15406
    ● Website: https://fatimaramone.github.io/
    Professional Summary:
    Fatima highlights her strengths as a web developer with "consistent code" and a persistent problem-solving attitude: "I don't sleep until I solve integration failures." She emphasizes her collaborative approach by mentioning her "family treatment," suggesting she values positive working relationships.
    Technical Skills:
    The resume lists a wide array of technical skills, demonstrating Fatima's proficiency in various programming languages, frameworks, and tools:
    ● Programming Languages: Python, JavaScript, Java, PHP
    ● Web Development: HTML5, CSS, Flask, Django, FastAPI
    ● Databases: SQL, SQLite, SQLAlchemy
    ● Other Tools: Git, Selenium, Langchain, Hugging Face, LLM's (Large Language Models), RAG (Retrieval Augmented Generation)
    Certificates:
    Fatima holds several professional certificates that further validate her skills and knowledge:
    ● IFC2010 DESARROLLO DE APLICACIONES CON TECNOLOGÍAS WEB: This 615-hour program focused on web application development and included modules like MF0491 (web programming in client environments), MF0492 (web programming in server environments), and MF0493 (implementation of web applications).
    ● IFCT 0109 SEGURIDAD INFORMÁTICA: This 420-hour program in computer security covered topics such as MF0223 (security in computer equipment), MF0226 (computer security auditing), and MF0227 (management of computer security incidents).
    Work Experience:
    Fatima's work experience is diverse, spanning web development, teaching, and event coordination.
    Web Development:
    ● 3.14 Financial Contents (April 2024 - May 2024): Backend Developer Intern. During her internship, Fatima was introduced to AI and machine learning concepts, including Langchain and other tools. She mentions gaining experience in REST, API (Application Programming Interface), Huggin Face, SOAP, and FastAPI.
    ● 2ksystems (September 2023 - October 2023): Full Stack Web Developer Intern. This internship provided Fatima with insights into the software industry and helped her develop significant JavaScript skills. Her responsibilities included working with HTML5, JSON, Web Services, and CSS.
    Other Experience:
    ● Fiverr (March 2020 - November 2022): TESOL Freelancer. Fatima worked remotely as a Spanish teacher and translator, primarily for American students. This role allowed her to refine her English skills and gain experience in online education and translation.
    ● Vice Media (February 2019 - March 2020): Event Coordinator at the "Old Blue Last" venue. Fatima coordinated events, collaborated with the marketing team, managed event logistics, and tracked attendance and feedback.
    ● The London Theatre (September 2016 - November 2018): Assistant Stage Manager. After starting as an intern, Fatima was promoted to a full-time role where she managed events, provided artist support, and coordinated theater projects and community programs.
    ● Barbican Center (April 2016 - September 2016): Assistant Stage Manager. Fatima worked on the production "Cathy Come Home," meeting and learning from the director, Ken Loach.
    ● The Brixton Jamm (December 2012 - February 2016): Event Coordinator. Fatima worked her way up from a barback to an event manager, gaining experience in staff training, stock control, and event organization.
    Education:
    Fatima's educational background combines formal degrees with specialized courses to enhance her skillset.
    ● Formal Education: Technical Superior Degree in Web Application Development from Rodolfo Ucha Piñeiro (September 2023 - Present).
    ● Informal Education:
    ○ Security Informatics (420 hours) (December 2023 - March 2024)
    ○ Development of Applications with Web Technologies (615 hours) (May 2023 - September 2023)
    ○ Object-Oriented Programming and Relational Databases (710 hours) (November 2022 - April 2023)
    Languages:
    ● English: Native fluency (10 years of residence in London).
    Projects:
    Fatima's resume includes descriptions of numerous projects, highlighting her practical application of technical skills.
    Web Development Projects:
    ● Flask Web App Básica: This basic web application built with Flask demonstrates Fatima's understanding of routing, HTML templates, and web application structure.
    ● Web Scraper de Productos con Selenium: Using Python and Selenium, this scraper extracts product information (name, price, discounts) from PCComponentes and saves it to a CSV file.
    ● Desktop\basic fastapi: This project involved building a simple web application using FastAPI to serve a static HTML file.
    ● ZALANDO: This project focused on building and deploying an image classification model using TensorFlow and the Fashion MNIST database.
    Other Projects:
    ● Android Application with Text Recognition and Database: This Android app uses ML Kit for text recognition, integrates a local database, and is built using Kotlin and the MVVM architecture.
    ● Automatización de Subida de Archivos HTML a Git: This PowerShell script automates the renaming and uploading of HTML files to a Git repository.
    ● Clasificación de Imágenes con TensorFlow y Flask: This project combines a TensorFlow-trained CNN model for image classification with a Flask-based HTTP server for predictions.
    ● Uso de la librería Konva.js en un Editor de Texto Enriquecido: This rich text editor utilizes Konva.js to render content on an interactive canvas and allows exporting to JPG format.
    ● Validación de Formulario en JavaScript: This script provides real-time form validation in JavaScript, checking for correct formats in various input fields.
    ● Android Manifest for Camera and Flashlight App: This Android manifest sets up permissions and app launch settings for an application using the camera and flashlight features.
    ● Galletitas Chinas de la Suerte: This HTML page generates random motivational quotes with interactive visual effects.
    Job Preferences:
    ● Desired Position: Web developer, preferably full-stack.
    ● Modality: Open to all work modalities.
    ● Location: A Coruña, Spain.
    ● Contract: Open to all contract types.
    ● Work Schedule: Open to all work schedules
    Other Information:
    ● Driving License: B
    ● Nationality: Spanish
    ● Work Permit: European Union
    ● Freelancer: No
    Overall Impression:
    Fatima Pita Perez's resume presents a strong candidate for web development positions. Her technical skills, diverse experience, project portfolio, and willingness to adapt to different work environments make her a valuable asset to any team. Her return to Spain and focus on A Coruña suggests a desire to establish her career in her home country.
    """
]


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
