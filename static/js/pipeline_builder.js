/**
 * MLForge - Interactive Visual Pipeline DAG Builder Controller
 */

const MLForgePipelineBuilder = {
    init() {
        const datasetSelect = document.getElementById('dataset_select');
        const targetInput = document.getElementById('target_select');
        const taskSelect = document.getElementById('task_select');
        const modelSelect = document.getElementById('model_select');

        if (datasetSelect) {
            datasetSelect.addEventListener('change', () => {
                const selectedOpt = datasetSelect.options[datasetSelect.selectedIndex];
                if (selectedOpt) {
                    const target = selectedOpt.getAttribute('data-target');
                    const task = selectedOpt.getAttribute('data-task');

                    if (targetInput && target) targetInput.value = target;
                    if (taskSelect && task) {
                        taskSelect.value = task;
                        this.updateModelOptions(task);
                    }
                }
            });
        }

        if (taskSelect) {
            taskSelect.addEventListener('change', () => {
                this.updateModelOptions(taskSelect.value);
            });
        }
    },

    updateModelOptions(taskType) {
        const modelSelect = document.getElementById('model_select');
        if (!modelSelect) return;

        const groups = modelSelect.querySelectorAll('optgroup');
        groups.forEach(group => {
            const label = group.getAttribute('label').toLowerCase();
            if (taskType === 'classification') {
                group.style.display = label.includes('classification') ? '' : 'none';
            } else {
                group.style.display = label.includes('regression') ? '' : 'none';
            }
        });
    }
};

document.addEventListener('DOMContentLoaded', () => {
    MLForgePipelineBuilder.init();
});

window.MLForgePipelineBuilder = MLForgePipelineBuilder;
