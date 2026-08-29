/**
 * MLForge - General Client-Side Utilities
 */

const MLForgeUtils = {
    /**
     * Formats a float to a percentage string.
     */
    formatPercent(value, decimals = 1) {
        if (value === null || value === undefined) return 'N/A';
        return `${(value * 100).toFixed(decimals)}%`;
    },

    /**
     * Formats a number with commas.
     */
    formatNumber(num) {
        if (num === null || num === undefined) return '0';
        return num.toLocaleString();
    },

    /**
     * Safe API fetch wrapper.
     */
    async fetchJson(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API Fetch Error [${url}]:`, error);
            throw error;
        }
    }
};

window.MLForgeUtils = MLForgeUtils;
