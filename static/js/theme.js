/**
 * MLForge - Theme Mode Switcher (Dark / Light)
 */

document.addEventListener('DOMContentLoaded', () => {
    const themeBtn = document.getElementById('themeToggleBtn');
    const themeIcon = document.getElementById('themeIcon');
    const htmlEl = document.documentElement;

    // Load saved theme or default to dark
    const savedTheme = MLForgeStorage.get('theme', 'dark');
    applyTheme(savedTheme);

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            const currentTheme = htmlEl.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            applyTheme(newTheme);
            MLForgeStorage.set('theme', newTheme);
        });
    }

    function applyTheme(theme) {
        htmlEl.setAttribute('data-theme', theme);
        if (themeIcon) {
            themeIcon.textContent = theme === 'dark' ? '🌙' : '☀️';
        }
    }
});
