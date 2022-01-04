from flask import Flask , render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def TROL():
    return render_template('index.html', prediction = request.form['Hour'])

app.run(host='127.0.0.1', port=8080, debug=True)