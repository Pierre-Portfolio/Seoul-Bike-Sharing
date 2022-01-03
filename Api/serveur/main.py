from flask import Flask , render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/trollAsterion/<ddd>/<ccc>")
def TROL(ddd, ccc):
    return render_template('index.html', asterion = "GG")

app.run(host='127.0.0.1', port=8080, debug=True)