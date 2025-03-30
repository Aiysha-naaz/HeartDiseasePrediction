function predict() {
    let data = {
        age: document.getElementById("age").value,
        gender: document.getElementById("gender").value,
        cholesterol: document.getElementById("cholesterol").value,
        blood_pressure: document.getElementById("blood_pressure").value,
        heart_rate: document.getElementById("heart_rate").value,
        diabetes: document.getElementById("diabetes").value,
        symptoms: document.getElementById("symptoms").value
    };

    fetch("/predict", {
        method: "POST",
        body: JSON.stringify(data),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById("result-message").textContent = result.message;
        
        if (result.status === "Disease Detected") {
            document.getElementById("disease").textContent = result.disease;
            document.getElementById("symptoms-entered").textContent = result.symptoms_entered.join(", ");
            document.getElementById("causes").textContent = result.causes.join(", ");
            document.getElementById("treatment").textContent = result.treatment.join(", ");
            document.getElementById("disease-details").style.display = "block";
        } else {
            document.getElementById("disease-details").style.display = "none";
        }
    })
    .catch(error => console.error("Error:", error));
}
