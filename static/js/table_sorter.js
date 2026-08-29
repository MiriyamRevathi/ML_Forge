/**
 * MLForge - Dynamic Table Sorter, Filter & Search Controller
 */

const MLForgeTable = {
    initSortableTables() {
        document.querySelectorAll('table.data-table[data-sortable="true"]').forEach(table => {
            const headers = table.querySelectorAll('th');
            headers.forEach((header, index) => {
                header.style.cursor = 'pointer';
                header.addEventListener('click', () => {
                    this.sortTable(table, index);
                });
            });
        });
    },

    sortTable(table, colIndex) {
        const tbody = table.querySelector('tbody');
        if (!tbody) return;

        const rows = Array.from(tbody.querySelectorAll('tr'));
        const isAscending = table.getAttribute('data-sort-asc') === 'true';

        rows.sort((a, b) => {
            const cellA = a.children[colIndex] ? a.children[colIndex].textContent.trim() : '';
            const cellB = b.children[colIndex] ? b.children[colIndex].textContent.trim() : '';

            const numA = parseFloat(cellA.replace(/[^0-9.-]+/g, ''));
            const numB = parseFloat(cellB.replace(/[^0-9.-]+/g, ''));

            if (!isNaN(numA) && !isNaN(numB)) {
                return isAscending ? numA - numB : numB - numA;
            }

            return isAscending ? cellA.localeCompare(cellB) : cellB.localeCompare(cellA);
        });

        rows.forEach(row => tbody.appendChild(row));
        table.setAttribute('data-sort-asc', !isAscending);
    },

    filterTable(searchInputId, tableId) {
        const searchInput = document.getElementById(searchInputId);
        const table = document.getElementById(tableId);

        if (!searchInput || !table) return;

        searchInput.addEventListener('keyup', () => {
            const query = searchInput.value.toLowerCase().trim();
            const rows = table.querySelectorAll('tbody tr');

            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(query) ? '' : 'none';
            });
        });
    }
};

document.addEventListener('DOMContentLoaded', () => {
    MLForgeTable.initSortableTables();
});

window.MLForgeTable = MLForgeTable;
