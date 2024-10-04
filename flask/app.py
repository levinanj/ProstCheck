from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    title = "Aplikasi Flask"
    message = "Ini adalah contoh menggunakan Flask dan Jinja2."
    return render_template('index.html', title=title, message=message)

if __name__ == '__main__':
    app.run(debug=True)
