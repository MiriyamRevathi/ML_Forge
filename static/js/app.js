/**
 * MLForge - Main Application Entry & Dynamic Inference Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log("⚡ MLForge ML Systems Platform Initialized.");

    // Dynamic Form Fields Generator for Predictions Page
    const modelSelect = document.getElementById('model_version_select');
    const dynamicFieldsContainer = document.getElementById('dynamicFormFields');
    const predictionForm = document.getElementById('predictionForm');
    const resultBox = document.getElementById('predictionResultBox');
    const resultContent = document.getElementById('predictionOutputContent');

    if (modelSelect && dynamicFieldsContainer) {
        modelSelect.addEventListener('change', async () => {
            const version = modelSelect.value;
            if (!version) return;

            try {
                const res = await MLForgeUtils.fetchJson(`/predictions/api/schema/${version}`);
                if (res.status === 'success') {
                    renderFormFields(res.feature_names);
                }
            } catch (err) {
                console.error("Failed to load model schema:", err);
            }
        });

        // Trigger initial load if value selected
        if (modelSelect.value) {
            modelSelect.dispatchEvent(new Event('change'));
        }
    }

    function renderFormFields(featureNames) {
        if (!featureNames || featureNames.length === 0) {
            dynamicFieldsContainer.innerHTML = '<p class="empty-cell">No features required for this model.</p>';
            return;
        }

        let html = '<div class="form-grid">';
        featureNames.forEach(feat => {
            html += `
                <div class="form-group">
                    <label for="feat_${feat}">${feat}</label>
                    <input type="text" id="feat_${feat}" name="${feat}" placeholder="Enter ${feat} value..." class="form-control" required>
                </div>
            `;
        });
        html += '</div>';
        dynamicFieldsContainer.innerHTML = html;
    }

    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(predictionForm);
            const modelVersion = formData.get('model_version');
            
            const inputData = {};
            formData.forEach((value, key) => {
                if (key !== 'model_version') {
                    // Try parsing numbers
                    const numVal = Number(value);
                    inputData[key] = !isNaN(numVal) && value.trim() !== '' ? numVal : value;
                }
            });

            try {
                const res = await MLForgeUtils.fetchJson('/predictions/api/predict', {
                    method: 'POST',
                    body: JSON.stringify({
                        model_version: modelVersion,
                        input_data: inputData
                    })
                });

                if (res.status === 'success') {
                    const output = res.result;
                    let displayStr = `Prediction: ${output.prediction}`;
                    if (output.probabilities) {
                        displayStr += ` (Probabilities: ${JSON.stringify(output.probabilities)})`;
                    }
                    resultContent.textContent = displayStr;
                    resultBox.style.display = 'block';
                    MLForgeToast.show('Inference executed successfully!', 'success');
                }
            } catch (err) {
                MLForgeToast.show(`Prediction Error: ${err.message}`, 'danger');
            }
        });
    }

    // Batch Prediction Form Handler
    const batchForm = document.getElementById('batchPredictionForm');
    if (batchForm) {
        batchForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(batchForm);

            try {
                const res = await fetch('/predictions/api/batch', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();

                if (data.status === 'success') {
                    const result = data.result;
                    MLForgeToast.show(`Batch Predictions Completed! Processed ${result.processed_rows} rows. Download link ready.`, 'success');
                    window.location.href = result.download_url;
                } else {
                    MLForgeToast.show(`Batch Error: ${data.message}`, 'danger');
                }
            } catch (err) {
                MLForgeToast.show(`Batch Upload Failed: ${err.message}`, 'danger');
            }
        });
    }
});
