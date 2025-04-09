import hashlib
from flask import Flask, jsonify, render_template, request, session, send_file, url_for
import pickle
import json
import pandas as pd
import os
from io import BytesIO
from fpdf import FPDF


from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend

import glob
import time

app = Flask(__name__)
app.secret_key = "your_secret_key"


@app.after_request
def remove_server_header(response):
    response.headers.pop('Server', None)  # Removes the Server header
    return response

# Load trained heart disease model
MODEL_PATH = "model/heart_disease_model.pkl"
INFO_PATH = "model/heart_disease_info.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(INFO_PATH):
    raise FileNotFoundError("Model or JSON file is missing. Check your file paths.")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(INFO_PATH, "r") as file:
    heart_disease_info_weighted = json.load(file)

 # Function to predict heart disease
def predict_heart_disease(age, gender, chol, bp, hr, diabetes, symptoms):
     gender_male = 1 if gender.lower() == "male" else 0
     diabetes_yes = 1 if diabetes.lower() == "yes" else 0

     input_data = pd.DataFrame([[age, chol, bp, hr, gender_male, diabetes_yes]], 
                               columns=["Age", "Cholesterol", "Blood Pressure", "Heart Rate", "Gender_Male", "Diabetes_Yes"])

     try:
         disease_present = model.predict(input_data)[0]
     except Exception as e:
         return {"status": "Error", "message": f"Model prediction failed: {str(e)}"}

     if disease_present == 0:
         return {
             "status": "Good Health",
             "message": "You are in good health! Keep it up! 😊",
         }
    
     return identify_heart_disease_type(symptoms)



def identify_heart_disease_type(symptoms):
    symptoms = [sym.strip().lower() for sym in symptoms]
    symptom_scores = {}

    for disease, details in heart_disease_info_weighted.items():
        total_score = 0
        matching_symptoms = {s.lower(): weight for s, weight in details["Symptoms"].items()}

        for sym in symptoms:
            total_score += matching_symptoms.get(sym, 0)

        symptom_scores[disease] = total_score

    max_score = max(symptom_scores.values())
    
    if max_score == 0:
        return {
            "status": "Uncertain",
            "message": "No matching disease detected.",
            "symptoms_entered": ", ".join(symptoms)
        }

    # Get all diseases with the max score
    best_matches = [d for d, score in symptom_scores.items() if score == max_score]

    # Return single or multiple diseases accordingly
    if len(best_matches) == 1:
       best = best_matches[0]
       return {
        "status": "Disease Detected",
        "disease": best,
        "causes": heart_disease_info_weighted[best]["Cause"],
        "treatment": heart_disease_info_weighted[best]["Treatment"],
        "symptoms_entered": ", ".join(symptoms),
        "message": f"Detected: {best}. Consult a doctor for further assistance."
    }
    else:
       results = []
    for disease in best_matches:
        results.append({
            "disease": disease,
            "cause": heart_disease_info_weighted[disease]["Cause"],
            "treatment": heart_disease_info_weighted[disease]["Treatment"]
        })

    return {
        "status": "Multiple Possibilities",
        "symptoms_entered": ", ".join(symptoms),
        "matched_diseases": results,
        "message": f"Detected multiple possible diseases. Please consult a doctor for accurate diagnosis."
    }



# Home route
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")





import glob
import plotly.graph_objects as go


import uuid
import matplotlib.pyplot as plt

# Function to clean up old graph files based on the age limit in seconds
def cleanup_old_graphs(directory, age_limit):
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.getmtime(file_path) < current_time - age_limit:
            os.remove(file_path)



def generate_graph(age, cholesterol, blood_pressure, heart_rate):
    # Data for the graph (replace these with the values you want)
    labels = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate']
    values = [age, cholesterol, blood_pressure, heart_rate]

    # Define custom colors for the bars
    colors = ['#19AADE', '#1AC9E6', '#6DF0D2', '#C7F9EE']  # Use your provided colors

    # Create a bar chart
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=colors)  # Apply the colors to the bars

    # Set graph title and labels
    ax.set_title("Patient's Health Metrics")
    ax.set_ylabel("Values")

    # Save the graph as a PNG file dynamically with a unique name
    graph_filename = f"graph_{time.time()}.png"  # Unique name based on timestamp
    graph_path = os.path.join('static/graphs', graph_filename)
    
    # Save the graph
    plt.savefig(graph_path, format='png')

    # Close the plot to avoid memory issues
    plt.close(fig)

    return graph_filename  # Return the graph file name for later use



@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Clean up old graph files before processing the request
    cleanup_old_graphs(directory="static/graphs", age_limit=3600)  # 1 hour

    if request.method == "POST":
        try:
            # Fetch input data from the form
            age = int(request.form["age"])
            gender = request.form["gender"]
            chol = int(request.form["cholesterol"])
            bp = int(request.form["blood_pressure"])
            hr = int(request.form["heart_rate"])
            diabetes = request.form["diabetes"]
            symptoms_text = request.form["symptoms"]

            # Clean and process symptoms
            symptoms = [sym.strip() for sym in symptoms_text.split(",") if sym.strip()]

            # Call prediction logic (replace with your actual implementation)
            result = predict_heart_disease(age, gender, chol, bp, hr, diabetes, symptoms)
           


            # Generate the graph dynamically and get the file name
            graph_file_name = generate_graph(age, chol, bp, hr)

            # Add patient details and graph info to the result
            result.update({
                "age": age,
                "gender": gender,
                "cholesterol": chol,
                "blood_pressure": bp,
                "heart_rate": hr,
                "diabetes": diabetes,
                "symptoms_entered": ", ".join(symptoms),
                "graph_file_name": graph_file_name  # Include the generated graph file name
            })
         
             # Save the result in the session for use in the report page
            session["result_summary"] = {
                "status": result.get("status"),
                "message": result.get("message"),
                "disease": result.get("disease", "---"),
                # "causes": result.get("causes", "No data available"),
                # "treatment": result.get("treatment", "No data available"),
                "causes": result.get("causes", []),  # Empty list if no causes
                "treatment": result.get("treatment", []),  # Empty list if no treatment
                "matched_diseases": result.get("matched_diseases", []),
                "age": result.get("age"),
                "gender": result.get("gender"),
                "cholesterol": result.get("cholesterol"),
                "blood_pressure": result.get("blood_pressure"),
                "heart_rate": result.get("heart_rate"),
                "diabetes": result.get("diabetes"),
                "symptoms_entered": result.get("symptoms_entered"),
                "graph_file_name": result.get("graph_file_name")  # Store graph info in the session
            }

            # Render the prediction page with the result
            return render_template("prediction.html", result=result)

        except Exception as e:
            # Handle errors and render an error page
            return render_template("error.html", error_message=f"Error occurred: {str(e)}")

    # Render the prediction page without results for GET requests
    return render_template("prediction.html", result=None)






@app.route("/report")
def report():
    result = session.get("result_summary", None)
    if not result:
        return render_template("error.html", error_message="No data available for the report. Please make a prediction first.")

    name = request.args.get("name", "Patient")
    print(name)
    return render_template("report.html", result=result, name=name)



from flask import render_template, session, make_response, jsonify, url_for
from io import BytesIO
from xhtml2pdf import pisa


import os
import base64



from flask import send_file  # if not already imported

@app.route("/download-pdf")
def download_pdf():
    result = session.get("result_summary", None)

    if not result:
        return jsonify({"error": "No report available to download."}), 400

    name = request.args.get("name", "Patient")
    try:
        # Get graph file path
        graph_filename = result.get("graph_file_name")
        graph_path = os.path.join("static", "graphs", graph_filename) if graph_filename else None

        # Convert graph image to Base64
        graph_base64 = None
        if graph_path and os.path.exists(graph_path):
            with open(graph_path, "rb") as img_file:
                graph_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Add timestamp
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d")

        # Add fallback for missing or empty causes/treatment
        result["causes"] = result.get("causes") or ["Not available"]
        result["treatment"] = result.get("treatment") or ["Not available"]

        # Render HTML template
        # html = render_template("report_pdf.html", result=result, graph_base64=graph_base64)
        html = render_template("report_pdf.html", result=result, graph_base64=graph_base64, name=name)


        # Generate PDF
        pdf = BytesIO()
        pisa_status = pisa.CreatePDF(html, pdf)

        if pisa_status.err:
            return jsonify({"error": "Error generating PDF"}), 500

        # Return downloadable PDF response
        response = make_response(pdf.getvalue())
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=Medical_Report.pdf"
        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":  
    app.run(debug=True)