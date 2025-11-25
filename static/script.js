/* script.js */

// Auto detect backend (local vs deployed)
const backendURL = window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:5000"
    : "https://life-insurance-prediction.onrender.com";  // <-- Replace if needed

document.getElementById("predictForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const age = document.getElementById("age").value;
    const gender = document.getElementById("gender").value;
    const income = document.getElementById("income").value;
    const health = document.getElementById("health").value;
    const smoke = document.getElementById("smoke").value;

    const res = await fetch(`${backendURL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ age, gender, income, health, smoke })
    });

    const data = await res.json();

    const result = document.getElementById("result");
    const policiesDiv = document.getElementById("policies");
    const premiumsDiv = document.getElementById("premiums");
    const suggestionsDiv = document.getElementById("suggestions");

    result.classList.remove("d-none");
    result.innerHTML = `<strong>${data.message}</strong>`;
    result.className = `alert ${data.eligible ? "alert-success" : "alert-danger"}`;

    if (data.eligible) {
        policiesDiv.innerHTML = `<strong>Policies:</strong> ${data.policies.join(", ")}`;

        let premiumHTML = "<strong>Premiums:</strong><br>";
        for (let p of data.policies) {
            premiumHTML += `${p}: ₹${data.premiums[p].toFixed(2)}<br>`;
        }
        premiumsDiv.innerHTML = premiumHTML;
    } else {
        policiesDiv.innerHTML = "";
        premiumsDiv.innerHTML = "";
    }

    suggestionsDiv.innerHTML = data.suggestions || "";
});
