#/mnt/c/Users/Admin/Downloads/aiweb22112024/aiweb-master$
from flask import Flask, send_from_directory
import os

app = Flask(__name__)

# Ruta para servir la página principal (index.html)
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Ruta para servir archivos estáticos de 'assets'
@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'assets'), filename)

# Ruta para servir archivos de 'Fonts'
@app.route('/Fonts/<path:filename>')
def fonts(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'Fonts'), filename)

# Ruta para servir archivos de 'cv'
@app.route('/cv/<path:filename>')
def cv_files(filename):
    return send_from_directory(os.path.join(os.getcwd(), 'cv'), filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
