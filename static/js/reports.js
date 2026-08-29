/**
 * MLForge - Platform Reports Viewer Controller
 */

const MLForgeReports = {
    init() {
        const copyBtn = document.getElementById('copyReportBtn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => {
                const markdownBody = document.querySelector('.markdown-body');
                if (markdownBody) {
                    navigator.clipboard.writeText(markdownBody.textContent).then(() => {
                        MLForgeToast.show('Report Markdown copied to clipboard!', 'success');
                    }).catch(err => {
                        MLForgeToast.show('Copy failed: ' + err.message, 'danger');
                    });
                }
            });
        }
    }
};

document.addEventListener('DOMContentLoaded', () => {
    MLForgeReports.init();
});

window.MLForgeReports = MLForgeReports;
