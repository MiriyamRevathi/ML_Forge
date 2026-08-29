/**
 * MLForge - Dashboard Event Handlers & Live Updates
 */

document.addEventListener('DOMContentLoaded', () => {
    // Mobile sidebar toggle handler
    const sidebarToggle = document.getElementById('sidebarToggle');
    const sidebar = document.getElementById('sidebar');

    if (sidebarToggle && sidebar) {
        sidebarToggle.addEventListener('click', () => {
            sidebar.classList.toggle('open');
        });
    }

    // Pipeline Builder Form Submit Handler
    const pipelineForm = document.getElementById('pipelineBuilderForm');
    if (pipelineForm) {
        pipelineForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(pipelineForm);
            const payload = {
                name: formData.get('name'),
                dataset_id: formData.get('dataset_id'),
                target_column: formData.get('target_column'),
                task: formData.get('task'),
                test_size: parseFloat(formData.get('test_size')),
                preprocessing: {
                    impute_strategy: formData.get('impute_strategy'),
                    scaler: formData.get('scaler'),
                    encoder: formData.get('encoder')
                },
                model: {
                    name: formData.get('model_name')
                }
            };

            try {
                const res = await MLForgeUtils.fetchJson('/pipelines/api/save', {
                    method: 'POST',
                    body: JSON.stringify(payload)
                });
                
                if (res.status === 'success') {
                    MLForgeToast.show('Pipeline DAG configuration saved successfully!', 'success');
                    setTimeout(() => {
                        window.location.href = '/pipelines/';
                    }, 1000);
                }
            } catch (err) {
                MLForgeToast.show(`Failed to save pipeline: ${err.message}`, 'danger');
            }
        });
    }
});
