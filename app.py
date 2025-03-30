import hashlib
from flask import Flask, jsonify, render_template, request, session, send_file, url_for
import pickle
import json
import pandas as pd
import os
from io import BytesIO
from fpdf import FPDF
#import plotly.express as px
#import plotly.io as pio

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


# Function to determine heart disease type based on symptoms
def identify_heart_disease_type(symptoms):
    symptoms = [sym.strip().lower() for sym in symptoms]  # Clean symptoms
    symptom_scores = {}

    for disease, details in heart_disease_info_weighted.items():
        total_score = sum(details["Symptoms"].get(sym, 0) for sym in symptoms)
        symptom_scores[disease] = total_score

    best_match = max(symptom_scores, key=symptom_scores.get)
    return {
        "status": "Disease Detected",
        "disease": best_match,
        "causes": list(heart_disease_info_weighted[best_match]["Cause"]),
        "treatment": list(heart_disease_info_weighted[best_match]["Treatment"]),
        "symptoms_entered": ", ".join(symptoms),
        "message": f"Detected: {best_match}. Consult a doctor for further assistance."
    }


# Home route
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")




# import plotly.express as px
# import pandas as pd
# from flask import Flask, render_template, request, session

# # Function to generate the graph
# def generate_graph(age, cholesterol, blood_pressure, heart_rate):
#     # Prepare data for the graph
#     data = {
#         "Parameter": ["Age", "Cholesterol", "Blood Pressure", "Heart Rate"],
#         "Value": [age, cholesterol, blood_pressure, heart_rate]
#     }
#     df = pd.DataFrame(data)

#     # Create a bar chart using Plotly
#     fig = px.bar(df, x="Parameter", y="Value", title="Health Parameters", color="Parameter")

#     # Return the graph as HTML (no need to save it, just render it)
#     return fig.to_html(full_html=False)


from datetime import datetime
import time
import glob
import plotly.graph_objects as go

import os
import time
import uuid
import matplotlib.pyplot as plt

# Function to clean up old graph files based on the age limit in seconds
def cleanup_old_graphs(directory, age_limit):
    current_time = time.time()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.getmtime(file_path) < current_time - age_limit:
            os.remove(file_path)

# # Function to generate a unique graph file name based on user inputs
# def generate_graph(age, chol, bp, hr):
#     # Use user inputs to create a unique string (e.g., a hash of the input)
#     user_input_string = f"{age}-{chol}-{bp}-{hr}"
#     unique_id = hashlib.md5(user_input_string.encode()).hexdigest()
    
#     # Define the path to save the graph image
#     graph_dir = "static/graphs"  # Folder to store the graphs
#     if not os.path.exists(graph_dir):
#         os.makedirs(graph_dir)
    
#     # Create a unique filename
#     graph_file_name = f"graph_{unique_id}.png"
#     graph_path = os.path.join(graph_dir, graph_file_name)

#     # Create the graph using matplotlib
#     plt.figure(figsize=(6, 4))
#     plt.title("Heart Disease Risk Factors")
#     plt.bar(["Age", "Cholesterol", "Blood Pressure", "Heart Rate"], [age, chol, bp, hr], color='skyblue')
#     plt.xlabel("Factors")
#     plt.ylabel("Values")
    
#     # Save the graph as a PNG image
#     plt.savefig(graph_path)
#     plt.close()

#     # Return the filename for rendering in the template
#     return graph_file_name


import matplotlib.pyplot as plt
import os
import time

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


# # Define your prediction route
# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":
#         try:
#             # Fetch input data from the form
#             age = int(request.form["age"])
#             gender = request.form["gender"]
#             chol = int(request.form["cholesterol"])
#             bp = int(request.form["blood_pressure"])
#             hr = int(request.form["heart_rate"])
#             diabetes = request.form["diabetes"]
#             symptoms_text = request.form["symptoms"]

#             # Clean and process symptoms
#             symptoms = [sym.strip() for sym in symptoms_text.split(",") if sym.strip()]

#             # Call prediction function (ensure this function is defined elsewhere in your code)
#             result = predict_heart_disease(age, gender, chol, bp, hr, diabetes, symptoms)

#             # Call the function to generate the graph
#             graph_html = generate_graph(age, chol, bp, hr)

#             # Add patient details to the result
#             result.update({
#                 "age": age,
#                 "gender": gender,
#                 "cholesterol": chol,
#                 "blood_pressure": bp,
#                 "heart_rate": hr,
#                 "diabetes": diabetes,
#                 "symptoms_entered": ", ".join(symptoms),
#                 "graph": graph_html
#             })

#             # Store a summary in the session to avoid large data in headers
#             session["result_summary"] = {
#                 "status": result.get("status"),
#                 "message": result.get("message"),
#                 "disease": result.get("disease", ""),
#                 "age": result.get("age"),
#                 "gender": result.get("gender"),
#                 "cholesterol": result.get("cholesterol"),
#                 "blood_pressure": result.get("blood_pressure"),
#                 "heart_rate": result.get("heart_rate"),
#                 "diabetes": result.get("diabetes"),
#                 "symptoms_entered": result.get("symptoms_entered")
#             }

#             return render_template("prediction.html", result=result)

#         except Exception as e:
#             return render_template("error.html", error_message=f"Error occurred: {str(e)}")

#     return render_template("prediction.html", result=None)



# # Define your prediction route
# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     cleanup_old_graphs() 
#     if request.method == "POST":
#         try:
#             # Fetch input data from the form
#             age = int(request.form["age"])
#             gender = request.form["gender"]
#             chol = int(request.form["cholesterol"])
#             bp = int(request.form["blood_pressure"])
#             hr = int(request.form["heart_rate"])
#             diabetes = request.form["diabetes"]
#             symptoms_text = request.form["symptoms"]

#             # Clean and process symptoms
#             symptoms = [sym.strip() for sym in symptoms_text.split(",") if sym.strip()]

#             # Call prediction function (ensure this function is defined elsewhere in your code)
#             result = predict_heart_disease(age, gender, chol, bp, hr, diabetes, symptoms)

#             # Call the function to generate the graph
#             graph_html = generate_graph(age, chol, bp, hr)

#             # Add patient details and other data to the result
#             result.update({
#                 "age": age,
#                 "gender": gender,
#                 "cholesterol": chol,
#                 "blood_pressure": bp,
#                 "heart_rate": hr,
#                 "diabetes": diabetes,
#                 "symptoms_entered": ", ".join(symptoms),
#                 "graph": graph_html
#             })

#             # Ensure causes and treatments are included in the session summary
#             session["result_summary"] = {
#                 "status": result.get("status"),
#                 "message": result.get("message"),
#                 "disease": result.get("disease", ""),
#                 "causes": result.get("causes", "No data available"),
#                 "treatment": result.get("treatment", "No data available"),
#                 "age": result.get("age"),
#                 "gender": result.get("gender"),
#                 "cholesterol": result.get("cholesterol"),
#                 "blood_pressure": result.get("blood_pressure"),
#                 "heart_rate": result.get("heart_rate"),
#                 "diabetes": result.get("diabetes"),
#                 "symptoms_entered": result.get("symptoms_entered"),
#                 "graph": result.get("graph")
#             }

#             return render_template("prediction.html", result=result)

#         except Exception as e:
#             return render_template("error.html", error_message=f"Error occurred: {str(e)}")

#     return render_template("prediction.html", result=None)




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


# # Define the report route
# @app.route("/report")
# def report():
#     result = session.get("result_summary", None)
#     if not result:
#         return render_template("error.html", error_message="No data available for the report. Please make a prediction first.")
#     return render_template("report.html", result=result)

# Define the report route
@app.route("/report")
def report():
    # Get result from session
    result = session.get("result_summary", None)
    if not result:
        return render_template("error.html", error_message="No data available for the report. Please make a prediction first.")

    return render_template("report.html", result=result)


# @app.route("/report")
# def report():
#     try:
#         result = session.get("result_summary", None)
#         print("Session Data Retrieved in /report:", result)  # Debugging output

#         if not result:
#             return render_template("error.html", error_message="No data available for the report. Please make a prediction first.")

#         return render_template("report.html", result=result)

#     except Exception as e:
#         print("Error in /report route:", str(e))  # Debugging output
#         return render_template("error.html", error_message=f"Error in /report route: {str(e)}")



# @app.route("/download-report")
# def download_report():
#     try:
#         # Get the result from the session
#         result = session.get("result_summary", None)
#         if not result:
#             return "No report available to download. Please complete a prediction first.", 400

#         # Generate PDF report
#         pdf = FPDF()
#         pdf.add_page()
#         pdf.set_font("Arial", size=12)

#         # Title
#         pdf.set_font("Arial", size=14, style="B")
#         pdf.cell(0, 10, "Heart Disease Prediction Report", ln=True, align="C")
#         pdf.ln(10)

#         # Add content to the PDF
#         pdf.set_font("Arial", size=12)
#         pdf.multi_cell(0, 10, f"""
#         Status: {result.get('status')}
#         Age: {result.get('age')}
#         Gender: {result.get('gender')}
#         Cholesterol: {result.get('cholesterol')}
#         Blood Pressure: {result.get('blood_pressure')}
#         Heart Rate: {result.get('heart_rate')}
#         Diabetes: {result.get('diabetes')}
#         Symptoms Entered: {result.get('symptoms_entered')}
#         Causes: {", ".join(result.get('causes', []))}
#         Treatment: {", ".join(result.get('treatment', []))}
#         """)
#         pdf.ln(10)

#         # Save the PDF to a byte stream
#         pdf_stream = BytesIO()
#         pdf.output(pdf_stream, 'F')
#         pdf_stream.seek(0)

#         # Return the PDF file
#         return send_file(pdf_stream, as_attachment=True, download_name="Heart_Disease_Report.pdf")

#     except Exception as e:
#         return f"Error generating report: {str(e)}", 500


# from flask import render_template, session, make_response, jsonify
# from io import BytesIO
# from xhtml2pdf import pisa

# @app.route("/download-pdf")
# def download_pdf():
#     result = session.get("result_summary", None)  # Fetch session data

#     if not result:
#         return jsonify({"error": "No report available to download."}), 400

#     try:
#         # Use report_pdf.html instead of report.html
#         html = render_template("report_pdf.html", result=result)

#         # Convert HTML to PDF
#         pdf = BytesIO()
#         pisa_status = pisa.CreatePDF(html, pdf)

#         if pisa_status.err:
#             return jsonify({"error": "Error generating PDF"}), 500

#         # Return the PDF as a downloadable file
#         response = make_response(pdf.getvalue())
#         response.headers["Content-Type"] = "application/pdf"
#         response.headers["Content-Disposition"] = "attachment; filename=Medical_Report.pdf"

#         return response

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


from flask import render_template, session, make_response, jsonify, url_for
from io import BytesIO
from xhtml2pdf import pisa
import os
# @app.route("/download-pdf")
# def download_pdf():
#     result = session.get("result_summary", None)  # Fetch session data

#     if not result:
#         return jsonify({"error": "No report available to download."}), 400

#     try:
#         # Fetch graph file name from session
#         graph_filename = result.get("graph_file_name", None)
        
#         if graph_filename:
#             graph_path = os.path.join("static/graphs", graph_filename)  # Construct full path
#         else:
#             graph_path = None

#         # Render the PDF template with the graph path
#         html = render_template("report_pdf.html", result=result, graph_path=graph_path)

#         # Convert HTML to PDF
#         pdf = BytesIO()
#         pisa_status = pisa.CreatePDF(html, pdf)

#         if pisa_status.err:
#             return jsonify({"error": "Error generating PDF"}), 500

#         # Return the PDF as a downloadable file
#         response = make_response(pdf.getvalue())
#         response.headers["Content-Type"] = "application/pdf"
#         response.headers["Content-Disposition"] = "attachment; filename=Medical_Report.pdf"

#         return response

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
import os
import base64

@app.route("/download-pdf")
def download_pdf():
    result = session.get("result_summary", None)

    if not result:
        return jsonify({"error": "No report available to download."}), 400

    try:
        # Get graph file path
        graph_filename = result.get("graph_file_name", None)
        graph_path = os.path.join("static", "graphs", graph_filename) if graph_filename else None

        # Convert image to Base64 (if graph exists)
        graph_base64 = None
        if graph_path and os.path.exists(graph_path):
            with open(graph_path, "rb") as img_file:
                graph_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        
        result["timestamp"] = datetime.now().strftime("%Y-%m-%d")        

        # Render PDF with base64-encoded image
        html = render_template("report_pdf.html", result=result, graph_base64=graph_base64)

        pdf = BytesIO()
        pisa_status = pisa.CreatePDF(html, pdf)

        if pisa_status.err:
            return jsonify({"error": "Error generating PDF"}), 500

        response = make_response(pdf.getvalue())
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = "attachment; filename=Medical_Report.pdf"

        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":  # Corrected the typo
    app.run(debug=True)