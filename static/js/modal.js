/**
 * MLForge - Modal Dialog & Overlay Manager
 */

const MLForgeModal = {
    open(modalId) {
        const overlay = document.getElementById(modalId);
        if (overlay) {
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    },

    close(modalId) {
        const overlay = document.getElementById(modalId);
        if (overlay) {
            overlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    },

    init() {
        document.querySelectorAll('[data-modal-target]').forEach(trigger => {
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = trigger.getAttribute('data-modal-target');
                this.open(targetId);
            });
        });

        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    this.close(overlay.id);
                }
            });
        });

        document.querySelectorAll('.modal-close-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const overlay = btn.closest('.modal-overlay');
                if (overlay) {
                    this.close(overlay.id);
                }
            });
        });
    }
};

document.addEventListener('DOMContentLoaded', () => {
    MLForgeModal.init();
});

window.MLForgeModal = MLForgeModal;
