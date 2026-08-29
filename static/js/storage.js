/**
 * MLForge - Local Storage State Persistence Manager
 */

const MLForgeStorage = {
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(`mlforge_${key}`);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error(`Error reading key ${key} from localStorage`, e);
            return defaultValue;
        }
    },

    set(key, value) {
        try {
            localStorage.setItem(`mlforge_${key}`, JSON.stringify(value));
            return true;
        } catch (e) {
            console.error(`Error writing key ${key} to localStorage`, e);
            return false;
        }
    },

    remove(key) {
        try {
            localStorage.removeItem(`mlforge_${key}`);
            return true;
        } catch (e) {
            return false;
        }
    }
};

window.MLForgeStorage = MLForgeStorage;
