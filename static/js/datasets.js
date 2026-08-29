/**
 * MLForge - Dataset Workspace Frontend Controller
 */

const MLForgeDatasets = {
    init() {
        const uploadForm = document.querySelector('.upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => {
                const fileInput = document.getElementById('file');
                if (fileInput && fileInput.files.length === 0) {
                    e.preventDefault();
                    MLForgeToast.show('Please select a CSV file to upload.', 'warning');
                }
            });
        }

        // Initialize sorting on dataset table
        const table = document.querySelector('.datasets-page table.data-table');
        if (table) {
            table.setAttribute('data-sortable', 'true');
        }
    }
};

document.addEventListener('DOMContentLoaded', () => {
    MLForgeDatasets.init();
});

window.MLForgeDatasets = MLForgeDatasets;
