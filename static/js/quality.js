/**
 * MLForge - Data Quality Audit Workspace Frontend Controller
 */

const MLForgeQuality = {
    init() {
        const filterInput = document.getElementById('qualityRuleSearch');
        if (filterInput) {
            MLForgeTable.filterTable('qualityRuleSearch', 'qualityRuleTable');
        }
    }
};

document.addEventListener('DOMContentLoaded', () => {
    MLForgeQuality.init();
});

window.MLForgeQuality = MLForgeQuality;
