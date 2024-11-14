AIWeb Chatbot

AIWeb Chatbot is a web application that integrates with the OpenAI API to provide chatbot responses using a Flask web server. The chatbot model uses LangChain to interact with OpenAI’s GPT-3.5, allowing for customizable responses. This project also leverages environment variables to securely handle API keys.
Features

    Interactive chatbot with customizable prompts.
    Environment variable support for secure API key handling.
    Flask-based backend for routing and handling user interactions.

Requirements

    Python 3.7+
    Flask
    LangChain
    OpenAI
    dotenv

Installation

    Clone the repository:

git clone https://github.com/FatimaRamone/aiweb.git
cd aiweb

Set up a virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Environment Variables:

    Create a .env file in the root directory.
    Add your OpenAI API key to the .env file:

        OPENAI_API_KEY=your_openai_api_key

Usage

    Run the Flask application:

    flask run

    Access the application: Open a web browser and go to http://127.0.0.1:5000 to interact with the chatbot.

File Structure

    app.py: The main Flask application file, containing routes and chatbot logic.
    templates/index.html: HTML front-end for interacting with the chatbot.
    .env: File to store environment variables securely.

Example

    Enter a question in the input field and submit it.
    The response will be displayed in the chatbot interface.

Security

The OpenAI API key is stored in the .env file and accessed through the dotenv library to avoid exposing sensitive information in the codebase.
Contributing

Feel free to open issues or submit pull requests for improvements and bug fixes.
License

This project is licensed under the MIT License.