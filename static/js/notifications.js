/**
 * MLForge - Client Toast Notification Controller
 */

const MLForgeToast = {
    show(message, type = 'info', duration = 4000) {
        const container = document.querySelector('.flash-messages-container');
        if (!container) return;

        const alert = document.createElement('div');
        alert.className = `alert alert-${type} alert-dismissible`;
        alert.innerHTML = `
            <span>${message}</span>
            <button type="button" class="alert-close" onclick="this.parentElement.remove()">×</button>
        `;

        container.appendChild(alert);

        setTimeout(() => {
            if (alert.parentNode) {
                alert.remove();
            }
        }, duration);
    }
};

window.MLForgeToast = MLForgeToast;
