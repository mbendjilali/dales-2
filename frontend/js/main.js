import { fetchTileList, fetchTileData, fetchTileVersions } from './api.js?v=15';
import { initScene, renderGraph, clearScene } from './viewer.js?v=15';
import { initInteraction } from './interaction.js?v=15';

async function initApp() {
    const container = document.body;
    initScene(container);
    
    // Populate tile selector
    const select = document.getElementById('tile-select');
    const versionHidden = document.getElementById('tile-version-id');
    try {
        const tiles = await fetchTileList();
        tiles.forEach(tileId => {
            const option = document.createElement('option');
            option.value = tileId;
            option.textContent = tileId;
            select.appendChild(option);
        });
    } catch (e) {
        console.error("Failed to load tiles", e);
    }

    async function loadTile(tileId, labelForStatus) {
        if (!tileId) {
            clearScene();
            document.getElementById('save-btn').disabled = true;
            return;
        }
        const display = labelForStatus || tileId;
        document.getElementById('status-msg').textContent = 'Loading...';
        try {
            const data = await fetchTileData(tileId);
            renderGraph(data);
            initInteraction(data); // Pass data to interaction
            document.getElementById('status-msg').textContent = `Loaded ${display}`;
            document.getElementById('save-btn').disabled = false;
        } catch (e) {
            console.error(e);
            document.getElementById('status-msg').textContent = 'Error loading tile';
        }
    }

    async function openVersionModal(baseId) {
        const modal = document.getElementById('version-modal');
        const baseSpan = document.getElementById('version-modal-base');
        const list = document.getElementById('version-list');
        const cancelBtn = document.getElementById('version-cancel-btn');
        if (!modal || !baseSpan || !list || !cancelBtn) {
            // Fallback: load original directly
            await loadTile(baseId);
            versionHidden.value = baseId;
            return;
        }

        baseSpan.textContent = baseId;
        list.innerHTML = '';

        try {
            const resp = await fetchTileVersions(baseId);
            const versions = Array.isArray(resp.versions) ? resp.versions : [];
            // If there's only one version, skip modal and load it directly
            if (versions.length === 1) {
                const only = versions[0];
                versionHidden.value = only.id;
                await loadTile(only.id, only.label);
                return;
            }

            versions.forEach(v => {
                const li = document.createElement('li');
                li.textContent = v.label || v.id;
                li.onclick = async () => {
                    modal.style.display = 'none';
                    versionHidden.value = v.id;
                    await loadTile(v.id, v.label);
                };
                list.appendChild(li);
            });

            cancelBtn.onclick = () => {
                modal.style.display = 'none';
                // Reset selection and hidden version if user cancels
                select.value = '';
                versionHidden.value = '';
                clearScene();
                document.getElementById('save-btn').disabled = true;
                document.getElementById('status-msg').textContent = '';
            };

            modal.style.display = 'flex';
        } catch (e) {
            console.error("Failed to load tile versions", e);
            // Fallback to original
            versionHidden.value = baseId;
            await loadTile(baseId);
        }
    }
    
    select.addEventListener('change', async (e) => {
        const baseId = e.target.value;
        if (!baseId) {
            clearScene();
            versionHidden.value = '';
            document.getElementById('save-btn').disabled = true;
            document.getElementById('status-msg').textContent = '';
            return;
        }
        // Reset current version; let the modal or loader set it
        versionHidden.value = '';
        await openVersionModal(baseId);
    });
}

initApp();
