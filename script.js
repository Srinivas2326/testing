/* script.js */
document.getElementById('predictForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    let age = document.getElementById('age').value;
    let gender = document.getElementById('gender').value;
    let income = document.getElementById('income').value;
    let health = document.getElementById('health').value;
    let smoke = document.getElementById('smoke').value;

    let response = await fetch('https://life-insurance-prediction.onrender.com', {  
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ age, gender, income, health, smoke })
    });

    let data = await response.json();

    let resultDiv = document.getElementById('result');
    resultDiv.classList.remove('d-none');

    resultDiv.innerHTML = `<strong>${data.message}</strong>`;
    resultDiv.classList.remove('alert-success', 'alert-danger');

    resultDiv.classList.add(data.eligible ? 'alert-success' : 'alert-danger');
    resultDiv.classList.add('animate__animated', 'animate__fadeIn');

    let policiesDiv = document.getElementById('policies');
    let premiumsDiv = document.getElementById('premiums');
    let suggestionsDiv = document.getElementById('suggestions');

    if (data.eligible) {
        policiesDiv.innerHTML = `<strong>Eligible Policies:</strong> ${data.policies.join(", ")}`;

        let premiumText = "<strong>Estimated Premiums:</strong><br>";

        for (let policy in data.premiums) {
            let value = data.premiums[policy];
            premiumText += `${policy}: ₹${value ? value.toFixed(2) : "N/A"}<br>`;
        }

        premiumsDiv.innerHTML = premiumText;
    } else {
        policiesDiv.innerHTML = "";
        premiumsDiv.innerHTML = "";
    }

    suggestionsDiv.innerHTML = data.suggestions ? `💡 ${data.suggestions}` : '';
});
