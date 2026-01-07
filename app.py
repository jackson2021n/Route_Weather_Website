from flask import Flask, render_template, request
from route_weather import build_town_weather_table

app = Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/route")
def route():
    start = request.form["start"].strip()
    dest = request.form["dest"].strip()
    depart = request.form["depart"].strip()

    meta, df = build_town_weather_table(start, dest, depart)
    table_html = df.to_html(index=False)

    return render_template("results.html", meta=meta, table_html=table_html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
