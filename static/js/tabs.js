/**
 * MLForge - Tab Switcher Controller
 */

const MLForgeTabs = {
    init() {
        document.querySelectorAll('.tab-group').forEach(group => {
            const tabs = group.querySelectorAll('.tab-btn');
            const contents = group.querySelectorAll('.tab-content');

            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    const targetId = tab.getAttribute('data-tab');

                    tabs.forEach(t => t.classList.remove('active'));
                    contents.forEach(c => c.classList.remove('active'));

                    tab.classList.add('active');
                    const targetContent = group.querySelector(`#${targetId}`);
                    if (targetContent) {
                        targetContent.classList.add('active');
                    }
                });
            });
        });
    }
};

document.addEventListener('DOMContentLoaded', () => {
    MLForgeTabs.init();
});

window.MLForgeTabs = MLForgeTabs;
