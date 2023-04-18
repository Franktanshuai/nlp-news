from flask import Flask

app = Flask(__name__)

@app.route("/")
def test():
    return "你好吗？"

if __name__ == "__main__":
    app.run()
