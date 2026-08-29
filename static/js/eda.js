/**
 * MLForge - Exploratory Data Analysis Frontend Controller
 */

const MLForgeEDA = {
    init() {
        document.querySelectorAll('.eda-page img').forEach(img => {
            img.style.cursor = 'pointer';
            img.addEventListener('click', () => {
                this.expandChartModal(img.src, img.alt);
            });
        });
    },

    expandChartModal(imageSrc, chartTitle) {
        let modal = document.getElementById('chartExpandModal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'chartExpandModal';
            modal.className = 'modal-overlay';
            modal.innerHTML = `
                <div class="modal-container modal-lg">
                    <div class="modal-header">
                        <h4 class="modal-title" id="expandedChartTitle">Chart View</h4>
                        <button class="modal-close-btn" onclick="MLForgeModal.close('chartExpandModal')">×</button>
                    </div>
                    <div class="modal-body" style="text-align: center;">
                        <img id="expandedChartImg" src="" style="max-width: 100%; height: auto; border-radius: var(--radius-md);">
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }

        document.getElementById('expandedChartTitle').textContent = chartTitle || 'Expanded Visualization';
        document.getElementById('expandedChartImg').src = imageSrc;
        MLForgeModal.open('chartExpandModal');
    }
};

document.addEventListener('DOMContentLoaded', () => {
    MLForgeEDA.init();
});

window.MLForgeEDA = MLForgeEDA;
