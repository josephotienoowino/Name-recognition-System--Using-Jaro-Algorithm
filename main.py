from flask import Flask, render_template, request
from jellyfish import jaro_winkler
import spacy
from jinja2 import Environment

app = Flask(__name__, template_folder='/home/jose/Desktop/Name Recognition Jaro-Winkler/template')

nlp = spacy.load("en_core_web_sm")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/results", methods=["POST"])
def results():
    # Get input text from form
    input_text = request.form.get("input_text")

    # Extract named entities from input text using spaCy
    doc = nlp(input_text)
    entities = [ent.text for ent in doc.ents]

    # Define reference list of known entities
    reference_list = ["Date","Time", "Person", "Location", "Organisation"]

    # Calculate Jaro-Winkler Distance score for each named entity
    scores = [jaro_winkler(entity, reference) for entity in entities for reference in reference_list]

    # Create a dictionary to store the named entities and their similarity scores
    results = {}
    for i, entity in enumerate(entities):
        for j, reference in enumerate(reference_list):
            key = f"{entity}_{j}"
            results[key] = scores[i * len(reference_list) + j]

    # Define Jinja2 environment
    env = Environment()

    # Render the results template and pass the results dictionary and environment to it
    return render_template("results.html", results=results, reference_list=reference_list, env=env)

if __name__ == "__main__":
    app.run(debug=True)
