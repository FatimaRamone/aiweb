# Document Chatbot

Este es un proyecto de chatbot que responde preguntas utilizando documentos locales como fuente de información. Está construido con **Flask**, **LangChain**, **OpenAI** y varias herramientas adicionales para procesamiento de texto.

## Características.

- Carga documentos desde archivos externos.
- Usa **TF-IDF** y **similaridad coseno** para encontrar documentos relevantes.
- Genera respuestas utilizando el modelo de OpenAI (GPT-3.5-turbo).
- Incluye una interfaz web sencilla para interactuar con el chatbot.
## Requisitos Previos

- Python 3.8 o superior.
- Clave API de OpenAI.
- Paquetes requeridos listados en `requirements.txt`.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/FatimaRamone/aiweb.git
   cd aiweb

    Crea un entorno virtual y actívalo:

python3 -m venv venv
source venv/bin/activate

Instala las dependencias:

pip install -r requirements.txt

Crea un archivo .env en el directorio raíz y añade tu clave API de OpenAI:

    OPENAI_API_KEY=tu_clave_api

    Asegúrate de que los documentos están en la carpeta static/documents/. Puedes agregar más archivos .txt según sea necesario.

Uso
Ejecutar con Gunicorn

Para iniciar el servidor en producción con Gunicorn:



Ejecutar en Desarrollo

Para iniciar el servidor Flask en modo de desarrollo:

python app.py

Accede a la aplicación en http://127.0.0.1:5000.
Estructura del Proyecto

aiweb/
│
├── app.py                # Código principal de la aplicación
├── requirements.txt      # Dependencias del proyecto
├── .env                  # Archivo para almacenar la clave API de OpenAI
├── templates/
│   └── index.html        # Interfaz HTML
├── static/
│   └── documents/        # Carpeta para documentos locales
│       └── documento.txt # Documento de ejemplo

Variables de Entorno

Asegúrate de configurar las siguientes variables en un archivo .env:

    OPENAI_API_KEY: Tu clave API de OpenAI.

Dependencias

Las dependencias necesarias están listadas en requirements.txt. Para instalarlas, usa:

pip install -r requirements.txt

Contribuir

    Haz un fork del repositorio.
    Crea una nueva rama para tu funcionalidad o corrección de errores:

git checkout -b mi-rama

Haz tus cambios y realiza un commit:

git commit -m "Añadida nueva funcionalidad"

Haz un push a tu rama:

    git push origin mi-rama

    Abre un Pull Request en GitHub.

Licencia

Este proyecto está licenciado bajo la MIT License.
