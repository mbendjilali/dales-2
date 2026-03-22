import * as THREE from 'three';
import { state } from './viewer.js?v=14';
import { editGroup, saveTile } from './api.js';

let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();

let lockedSelection = null; // { type, id }
let previewSelection = null; // { type, id }
let multiSelection = []; // Array of { type, id } for multi-select

// Colors (matching viewer.js/python script)
const COLOR_SELECTED = 0xffff00; 
const COLOR_EXTENSION = 0xffa500; 
const COLOR_BIFURCATION = 0x800080; 
const COLOR_CROSS = 0xff0000; 
const COLOR_SUPPORT = 0x1565c0; 
// Slightly higher dim opacity to keep context more visible
const OPACITY_DIM = 0.5;
const OPACITY_DIM_POLE = 0.25;

let currentData = null;

function _buildGroupMembershipLookup(groups) {
    const lookup = new Map();
    if (!Array.isArray(groups)) return lookup;
    groups.forEach(g => {
        const members = g.members || [];
        members.forEach(mid => lookup.set(mid, { gid: g.id, members }));
    });
    return lookup;
}
function _getGroupedIds(type, id) {
    const groupsKey = { building: 'buildingGroups', vehicle: 'vehicleGroups', tree: 'treeGroups' }[type];
    if (!groupsKey || !currentData) return [];
    const groups = currentData[groupsKey] || [];
    for (const g of groups) {
        if ((g.members || []).includes(id)) {
            return g.members.filter(m => m !== id);
        }
    }
    return [];
}

// Helper for semantic class names
function getSemanticClassName(type, sem_class) {
    if (sem_class === undefined || sem_class === null) {
        return type.charAt(0).toUpperCase() + type.slice(1);
    }
    const map = {
        0: 'Ground',
        1: 'Vegetation',
        2: 'Car',
        3: 'Powerline',
        4: 'Fence',
        5: 'Tree',
        6: 'Pick up',
        7: 'Van & Truck',
        8: 'Heavy-duty',
        9: 'Utility pole',
        10: 'Light pole',
        11: 'Traffic pole',
        12: 'Habitat',
        13: 'Complex',
        14: 'Annex'
    };
    return map[sem_class] || type.charAt(0).toUpperCase() + type.slice(1);
}

// In-memory index for typed group relations: member_type -> { byGroup: Map(gid -> [rels]) }
const groupRelationsIndex = {
    building: { byGroup: new Map() },
    vehicle: { byGroup: new Map() },
    tree: { byGroup: new Map() },
};

const RELATION_PRIORITY = ["adjacent", "near"];

function buildGroupRelationsIndex(relations) {
    // Reset
    Object.keys(groupRelationsIndex).forEach(mt => {
        groupRelationsIndex[mt].byGroup.clear();
    });
    if (!Array.isArray(relations)) return;
    relations.forEach(r => {
        const mt = r.member_type;
        if (!groupRelationsIndex[mt]) return;
        let gid = String(r.group_id);
        if (mt === 'tree' && r.peer_type) {
            gid = '0';
        }
        const bucket = groupRelationsIndex[mt].byGroup;
        let arr = bucket.get(gid);
        if (!arr) {
            arr = [];
            bucket.set(gid, arr);
        }
        arr.push(r);
    });
}

// Visibility state per class
const visibilityState = {
    conductors: true,
    poles: true,
    buildings: true,
    vehicles: true,
    trees: true,
};

function macroClassLabel(macro) {
    const cls = macro.user_class || macro.type || "";
    const map = {
        connector_span: "Connector span",
        electrical_grid: "Electrical grid",
    };
    return map[cls] || cls || "Macro";
}

function buildMacroInstancesFromData(data) {
    const macros = [];
    (data.connector_spans || []).forEach(s => {
        macros.push({
            id: s.id, type: 'connector_span', member_type: 'conductor',
            member_ids: s.conductor_ids || [], user_class: 'connector_span',
            label: s.label || '', auto: true, metadata: { support_poles: s.poles || [] },
        });
    });
    (data.electrical_grids || []).forEach(g => {
        macros.push({
            id: g.id, type: 'electrical_grid', member_type: 'connector_span',
            member_ids: g.span_ids || [], user_class: 'electrical_grid',
            label: g.label || '', auto: true, metadata: {},
        });
    });
    return macros;
}

// In-memory reverse index: member_type -> Map(member_id_string -> [macros])
let macroMembershipIndex = {};

function buildMacroMembershipIndex(macros) {
    const index = {};
    if (!Array.isArray(macros)) return index;
    macros.forEach(m => {
        const memberType = m.member_type;
        if (!memberType || !Array.isArray(m.member_ids)) return;
        if (!index[memberType]) {
            index[memberType] = new Map();
        }
        const map = index[memberType];
        m.member_ids.forEach(id => {
            const key = String(id);
            let list = map.get(key);
            if (!list) {
                list = [];
                map.set(key, list);
            }
            list.push(m);
        });
    });
    return index;
}

// --- Color scheme state (semantic vs per-instance vs macro-based) ---
let currentColorScheme = 'semantic'; // 'semantic' | 'instance' | 'macro'
const instanceColors = {
    conductor: new Map(),
    pole: new Map(),
    building: new Map(),
    vehicle: new Map(),
    // Trees are handled via instancing; left semantic for now
};

const _instanceTempColor = new THREE.Color();
// Macro-based colors: one color per macro id
const macroColors = new Map();

function colorFromIdStable(id) {
    // Deterministic hue from id (works for numbers and strings)
    const str = String(id);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = (hash * 31 + str.charCodeAt(i)) >>> 0;
    }
    const h = (hash % 360) / 360; // [0,1)
    _instanceTempColor.setHSL(h, 0.6, 0.5);
    return _instanceTempColor.getHex();
}

function colorRandom() {
    const h = Math.random();
    _instanceTempColor.setHSL(h, 0.6, 0.5);
    return _instanceTempColor.getHex();
}

function clearInstanceColors() {
    Object.keys(instanceColors).forEach(key => {
        instanceColors[key].clear();
    });
}

function clearMacroColors() {
    macroColors.clear();
}

function buildInstanceColorsFromData(data, randomize = false) {
    if (!data) return;
    clearInstanceColors();

    const pick = randomize ? colorRandom : colorFromIdStable;

    if (Array.isArray(data.conductors)) {
        const map = instanceColors.conductor;
        data.conductors.forEach(c => {
            if (c && c.id != null) {
                const colorKey = c.id != null ? `inst:${c.id}` : 'conductor';
                map.set(c.id, pick(colorKey));
            }
        });
    }

    if (Array.isArray(data.poles)) {
        const map = instanceColors.pole;
        data.poles.forEach(p => {
            if (p && p.id != null) {
                map.set(p.id, pick(p.id));
            }
        });
    }

    if (Array.isArray(data.buildings)) {
        const map = instanceColors.building;
        data.buildings.forEach(b => {
            if (b && b.id != null) {
                map.set(b.id, pick(b.id));
            }
        });
    }

    if (Array.isArray(data.vehicles)) {
        const map = instanceColors.vehicle;
        data.vehicles.forEach(v => {
            if (v && v.id != null) {
                map.set(v.id, pick(v.id));
            }
        });
    }
}

function randomizeInstanceColors(data) {
    buildInstanceColorsFromData(data, true);
}

function getInstanceColor(typeKey, id, fallback) {
    const map = instanceColors[typeKey];
    if (!map) return fallback;
    const val = map.get(id);
    return (typeof val === 'number') ? val : fallback;
}

function buildMacroColorsFromMacros(macros, randomize = false) {
    clearMacroColors();
    if (!Array.isArray(macros)) return;
    const pick = randomize ? colorRandom : colorFromIdStable;
    macros.forEach(m => {
        const key = `${m.type || 'macro'}:${m.id}`;
        macroColors.set(m.id, pick(key));
    });
}

function _groupIdForClusterObject(groupsKey, objectId) {
    const groups = currentData && currentData[groupsKey];
    if (!Array.isArray(groups)) return null;
    const oid = Number(objectId);
    for (const g of groups) {
        const mem = g.members || [];
        if (mem.some(m => Number(m) === oid)) return g.id;
    }
    return null;
}

function getMacroColor(typeKey, id, fallback) {
    if (!currentData) return fallback;
    // Map internal color type keys to macro member_type
    const memberTypeByKey = {
        conductor: 'conductor',
        pole: 'pole',
        building: 'building',
        vehicle: 'vehicle',
        tree: 'tree',
    };
    const memberType = memberTypeByKey[typeKey];
    if (!memberType) return fallback;

    if (macroMembershipIndex) {
        const typeMap = macroMembershipIndex[memberType];
        if (typeMap) {
            const list = typeMap.get(String(id));
            if (list && list.length) {
                const macro = list[0];
                if (!macroColors.size) {
                    buildMacroColorsFromMacros(currentData.macro_instances || []);
                }
                const val = macroColors.get(macro.id);
                if (typeof val === 'number') return val;
            }
        }
    }

    // Building / vehicle / tree clusters: color by macro tab group_id (no electrical macro)
    const clusterKey = {
        building: 'buildingGroups',
        vehicle: 'vehicleGroups',
        tree: 'treeGroups',
    }[typeKey];
    if (clusterKey) {
        const gid = _groupIdForClusterObject(clusterKey, id);
        if (gid != null && gid !== undefined) {
            return colorFromIdStable(`cluster:${typeKey}:${gid}`);
        }
    }

    return fallback;
}

// Global helper: find all macro instances related to a selection
function getMacrosForSelection(sel) {
    const data = currentData;
    const macros = Array.isArray(data && data.macro_instances) ? data.macro_instances : [];
    const out = [];
    if (!macros.length || !sel) return out;

    if (sel.type === 'macro') {
        const m = macros.find(m => m.id === sel.id && m.type === sel.macroType);
        if (m) out.push(m);
        return out;
    }

    if (!macroMembershipIndex) return out;

    if (sel.type === 'conductor') {
        const cidKey = String(sel.id);
        const conductorMap = macroMembershipIndex['conductor'];
        const spans = conductorMap ? (conductorMap.get(cidKey) || []) : [];
        spans.forEach(m => out.push(m));

        // For each span, find grids that include this span as a member (member_type = connector_span)
        const spanIds = new Set(spans.map(m => String(m.id)));
        const spanMacroMap = macroMembershipIndex['connector_span'];
        if (spanMacroMap) {
            spanIds.forEach(spanId => {
                const gridsForSpan = spanMacroMap.get(spanId) || [];
                gridsForSpan.forEach(g => out.push(g));
            });
        }
        return out;
    }

    const typeBySelection = {
        building: 'building',
        vehicle: 'vehicle',
        tree: 'tree',
        pole: 'pole',
    };
    const memberType = typeBySelection[sel.type];
    if (memberType && macroMembershipIndex[memberType]) {
        const map = macroMembershipIndex[memberType];
        const list = map.get(String(sel.id)) || [];
        list.forEach(m => out.push(m));
    }
    return out;
}

// Lasso selection state
let lassoActive = false;
let lassoPending = false;
let lassoStart = null;
let lassoEnd = null;
let lassoEl = null;

export function initInteraction(data) {
    currentData = data;
    const macros = buildMacroInstancesFromData(data);
    data.macro_instances = macros;
    state.macro.all = macros;
    state.macro.connectorSpans = macros.filter(m => m.type === 'connector_span');
    state.macro.electricalGrids = macros.filter(m => m.type === 'electrical_grid');
    state.macro.buildingGroups = data.buildingGroups || [];
    state.macro.vehicleGroups = data.vehicleGroups || [];
    state.macro.treeGroups = data.treeGroups || [];
    macroMembershipIndex = buildMacroMembershipIndex(macros);
    buildGroupRelationsIndex(data.groupRelations || []);
    // Reset per-instance and macro colors for this tile (rebuilt lazily on first use)
    clearInstanceColors();
    clearMacroColors();
    const canvas = state.renderer.domElement;
    
    canvas.addEventListener('click', onClick);
    canvas.addEventListener('dblclick', onDoubleClick);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerdown', onPointerDownLasso);
    window.addEventListener('pointerup', onPointerUpLasso);
    window.addEventListener('keydown', onKeyDown);
    
    // UI Event Listeners
    document.getElementById('save-btn').onclick = async () => {
        const versionEl = document.getElementById('tile-version-id');
        const tileId = (versionEl && versionEl.value) || document.getElementById('tile-select').value;
        if (tileId) {
            document.getElementById('status-msg').textContent = 'Saving...';
            try {
                await saveTile(tileId);
                document.getElementById('status-msg').textContent = 'Saved!';
                setTimeout(() => document.getElementById('status-msg').textContent = '', 2000);
            } catch (e) {
                console.error(e);
                document.getElementById('status-msg').textContent = 'Error saving';
            }
        }
    };

    const uploadBtn = document.getElementById('upload-btn');
    const uploadModal = document.getElementById('upload-modal');
    const uploadCancel = document.getElementById('upload-cancel-btn');
    const uploadProcess = document.getElementById('upload-process-btn');
    if (uploadBtn && uploadModal) {
        uploadBtn.onclick = () => { uploadModal.style.display = 'flex'; };
        uploadCancel.onclick = () => { uploadModal.style.display = 'none'; };
        uploadProcess.onclick = async () => {
            const lazInput = document.getElementById('upload-laz-input');
            const netInput = document.getElementById('upload-net-input');
            const statusEl = document.getElementById('upload-status');
            if (!lazInput.files.length) { statusEl.textContent = 'Please select a LAZ/LAS file.'; return; }
            statusEl.textContent = 'Processing...';
            const fd = new FormData();
            fd.append('laz_file', lazInput.files[0]);
            if (netInput.files.length) fd.append('network_file', netInput.files[0]);
            try {
                const resp = await fetch('/api/load_laz', { method: 'POST', body: fd });
                const result = await resp.json();
                if (resp.ok && result.tile_id) {
                    statusEl.textContent = `Done: tile ${result.tile_id}`;
                    uploadModal.style.display = 'none';
                    const sel = document.getElementById('tile-select');
                    const opt = document.createElement('option');
                    opt.value = result.tile_id;
                    opt.textContent = result.tile_id;
                    sel.appendChild(opt);
                    sel.value = result.tile_id;
                    sel.dispatchEvent(new Event('change'));
                } else {
                    statusEl.textContent = `Error: ${result.detail || 'unknown'}`;
                }
            } catch (err) {
                statusEl.textContent = `Error: ${err.message}`;
            }
        };
    }

    document.getElementById('openListBtn').onclick = () => {
        const el = document.getElementById('sidebar');
        el.style.display = (el.style.display === 'none') ? 'flex' : 'none';
    };

    document.getElementById('closeSidebarBtn').onclick = () => {
        document.getElementById('sidebar').style.display = 'none';
    };
    
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.onclick = () => {
            const tabName = btn.dataset.tab;
            openTab(tabName);
        };
    });

    // Macro secondary tabs
    document.querySelectorAll('.macro-tab-btn').forEach(btn => {
        btn.onclick = () => {
            const name = btn.dataset.macroTab;
            if (name) openMacroSubTab(name);
        };
    });
    
    document.getElementById('wedge-btn').onclick = () => handleEdit('wedge');
    document.getElementById('split-btn').onclick = () => handleEdit('split');
    document.getElementById('delete-btn').onclick = () => handleEdit('delete');

    // Color scheme toggle
    const colorToggle = document.getElementById('color-scheme-toggle');
    if (colorToggle) {
        const refreshLabel = () => {
            if (currentColorScheme === 'semantic') {
                colorToggle.textContent = 'Semantic colors';
            } else if (currentColorScheme === 'instance') {
                colorToggle.textContent = 'Instance colors';
            } else {
                colorToggle.textContent = 'Macro colors';
            }
        };
        refreshLabel();
        colorToggle.onclick = () => {
            toggleColorScheme();
            refreshLabel();
            updateHighlights();
        };
    }

    // Visibility toggle buttons
    document.querySelectorAll('#visibility-bar .vis-toggle').forEach(btn => {
        const cls = btn.getAttribute('data-class');
        if (!cls) return;
        // Initial state
        btn.classList.toggle('off', !visibilityState[cls]);
        btn.onclick = () => {
            toggleVisibility(cls);
        };
    });

    applyVisibility();
    
    // Populate Lists
    populateLists(data);
}

function applyVisibility() {
    // Apply visibility to meshes
    state.meshes.conductors.forEach(m => { m.visible = visibilityState.conductors; });
    state.meshes.poles.forEach(m => { m.visible = visibilityState.poles; });
    state.meshes.buildings.forEach(m => { m.visible = visibilityState.buildings; });
    state.meshes.vehicles.forEach(m => { m.visible = visibilityState.vehicles; });
    if (state.meshes.treeMesh) {
        state.meshes.treeMesh.visible = visibilityState.trees;
    }
    // Tree highlights follow tree visibility
    if (state.meshes.treeHighlights && state.meshes.treeHighlights.length) {
        state.meshes.treeHighlights.forEach(m => { m.visible = visibilityState.trees; });
    }
}

const TYPE_TO_VIS_KEY = {
    conductor: 'conductors',
    pole: 'poles',
    building: 'buildings',
    vehicle: 'vehicles',
    tree: 'trees',
};

function isTypeVisible(type) {
    const key = TYPE_TO_VIS_KEY[type];
    if (!key) return true;
    return !!visibilityState[key];
}

function toggleVisibility(clsKey) {
    if (!(clsKey in visibilityState)) return;
    visibilityState[clsKey] = !visibilityState[clsKey];

    // Update button state
    const btn = document.querySelector(`#visibility-bar .vis-toggle[data-class="${clsKey}"]`);
    if (btn) {
        btn.classList.toggle('off', !visibilityState[clsKey]);
    }

    applyVisibility();

    // Drop any selection/preview for now-hidden types
    const isKeyVisible = visibilityState[clsKey];
    if (!isKeyVisible) {
        const affectedType = Object.entries(TYPE_TO_VIS_KEY)
            .find(([, v]) => v === clsKey)?.[0];
        if (affectedType) {
            multiSelection = multiSelection.filter(s => s.type !== affectedType);
            if (lockedSelection && lockedSelection.type === affectedType) {
                lockedSelection = null;
            }
            if (previewSelection && previewSelection.type === affectedType) {
                previewSelection = null;
            }
        }
    }

    updateHighlights();
    updateActionBar();
    updateInfoPanel(lockedSelection);
}

function onKeyDown(e) {
    const tag = (e.target && e.target.tagName) ? e.target.tagName.toUpperCase() : '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    const key = e.key || '';
    const lower = key.toLowerCase();

    if (lower === 'w') {
        e.preventDefault();
        handleEdit('wedge');
        return;
    }
    if (lower === 'x') {
        e.preventDefault();
        handleEdit('split');
        return;
    }
    if (lower === 'c') {
        e.preventDefault();
        handleEdit('delete');
        return;
    }

    // Color scheme toggles: Space and Shift+Space
    if (e.code === 'Space') {
        e.preventDefault();
        if (e.shiftKey) {
            // Randomize palettes without changing the current scheme
            randomizeInstanceColors(currentData);
            if (currentColorScheme === 'macro') {
                buildMacroColorsFromMacros(currentData.macro_instances || [], true);
            }
        } else {
            const colorToggle = document.getElementById('color-scheme-toggle');
            const refreshLabel = () => {
                if (currentColorScheme === 'semantic') {
                    colorToggle.textContent = 'Semantic colors';
                } else if (currentColorScheme === 'instance') {
                    colorToggle.textContent = 'Instance colors';
                } else {
                    colorToggle.textContent = 'Macro colors';
                }
            };
            refreshLabel();
            toggleColorScheme();
        }
        updateHighlights();
        return;
    }

    if (lower === 'g') {
        e.preventDefault();
        handleGroupRelationCycleShortcut();
        return;
    }

    // Visibility shortcuts: A, Z, E, R, T
    const visKeyByChar = {
        a: 'conductors',
        z: 'poles',
        e: 'buildings',
        r: 'vehicles',
        t: 'trees',
    };
    const visKey = visKeyByChar[lower];
    if (visKey) {
        e.preventDefault();
        toggleVisibility(visKey);
        return;
    }
}

function ensureLassoElement() {
    if (!lassoEl) {
        lassoEl = document.getElementById('lasso-rect');
    }
}

function onPointerDownLasso(event) {
    // Start potential lasso on Shift + left button
    if (!event.shiftKey || event.button !== 0) return;
    lassoPending = true;
    lassoActive = false;
    ensureLassoElement();
    if (!lassoEl) return;
    lassoStart = { x: event.clientX, y: event.clientY };
    lassoEnd = { x: event.clientX, y: event.clientY };
    // Do not draw yet; wait for movement threshold in onPointerMove
    event.preventDefault();
}

function onPointerUpLasso(event) {
    const hadPending = lassoPending;
    const hadActive = lassoActive;
    lassoPending = false;
    lassoActive = false;
    ensureLassoElement();
    if (lassoEl) {
        lassoEl.style.display = 'none';
    }
    // Enable camera interaction
    if (state.controls) {
        state.controls.enabled = true;
    }
    // Only apply selection if we actually dragged a lasso
    if (hadActive) {
        applyLassoSelection();
    }
}

function updateLassoVisual() {
    if (!lassoEl || !lassoStart || !lassoEnd) return;
    const x1 = lassoStart.x;
    const y1 = lassoStart.y;
    const x2 = lassoEnd.x;
    const y2 = lassoEnd.y;
    const left = Math.min(x1, x2);
    const top = Math.min(y1, y2);
    const width = Math.abs(x2 - x1);
    const height = Math.abs(y2 - y1);
    lassoEl.style.display = 'block';
    lassoEl.style.left = `${left}px`;
    lassoEl.style.top = `${top}px`;
    lassoEl.style.width = `${width}px`;
    lassoEl.style.height = `${height}px`;
}

const _tempVec3 = new THREE.Vector3();
const _lassoBox3 = new THREE.Box3();

// Tree highlighting via instanced meshes (base / selected / group)
let lastTreeSelectionKey = null;

function rebuildTreeMeshes(selectedTreeIdsSet, groupedTreeIdsSet) {
    if (!currentData || !currentData.trees) return;

    // Build stable key so we don't rebuild unnecessarily
    const selArr = Array.from(selectedTreeIdsSet).sort((a, b) => a - b);
    const grpArr = Array.from(groupedTreeIdsSet).sort((a, b) => a - b);
    const key = JSON.stringify({ sel: selArr, grp: grpArr });
    if (key === lastTreeSelectionKey) return;
    lastTreeSelectionKey = key;

    // Remove existing tree meshes (we no longer use treeHighlights overlays)
    if (state.meshes.trees && state.meshes.trees.length) {
        state.meshes.trees.forEach(m => {
            state.scene.remove(m);
            if (m.geometry) m.geometry.dispose();
            if (m.material) {
                if (Array.isArray(m.material)) m.material.forEach(mat => mat.dispose());
                else m.material.dispose();
            }
        });
    }
    state.meshes.trees = [];
    state.meshes.treeMesh = null;
    state.meshes.treeHighlights = [];

    const trees = currentData.trees;
    if (!trees.length) return;

    const selectedSet = new Set(selectedTreeIdsSet);
    const groupedSet = new Set(groupedTreeIdsSet);

    const baseTrees = [];
    const selectedTrees = [];
    const groupedTrees = [];

    for (const t of trees) {
        if (selectedSet.has(t.id)) {
            selectedTrees.push(t);
        } else if (groupedSet.has(t.id)) {
            groupedTrees.push(t);
        } else {
            baseTrees.push(t);
        }
    }

    function makeTreeMesh(treeList, color) {
        if (!treeList.length) return null;
        const treeGeom = new THREE.ConeGeometry(1.0, 1.0, 10);
        treeGeom.rotateX(Math.PI / 2);
        const treeMat = new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.8,
        });
        const count = treeList.length;
        const instanced = new THREE.InstancedMesh(treeGeom, treeMat, count);
        const dummy = new THREE.Object3D();
        const ids = [];
        treeList.forEach((t, index) => {
            const x = t.X ?? 0;
            const y = t.Y ?? 0;
            const z0 = t.Z ?? 0;
            const h = t.height ?? 1.0;
            const r = t.crown_radius ?? 1.0;

            dummy.position.set(x, y, z0);
            dummy.scale.set(r, r, h);
            dummy.rotation.set(0, 0, 0);
            dummy.updateMatrix();

            instanced.setMatrixAt(index, dummy.matrix);
            ids[index] = t.id;
        });

        instanced.userData.type = 'tree';
        instanced.userData.ids = ids;
        state.scene.add(instanced);
        state.meshes.trees.push(instanced);
        return instanced;
    }

    const baseMesh = makeTreeMesh(baseTrees, 0x228b22);
    const selectedMesh = makeTreeMesh(selectedTrees, COLOR_SELECTED);
    const groupMesh = makeTreeMesh(groupedTrees, COLOR_EXTENSION);

    state.meshes.treeMesh = baseMesh || selectedMesh || groupMesh || null;
}

/**
 * Project world AABB to screen; returns { left, right, top, bottom } in client (viewport) coords.
 * Only uses corners in front of the camera (NDC z <= 1) so objects behind don't get huge AABBs.
 * Uses the canvas bounding rect so coordinates match lasso (clientX/clientY).
 * Result is clamped to the canvas rect so we only consider the visible footprint.
 */
function getScreenAABB(worldMin, worldMax) {
    const canvas = state.renderer.domElement;
    const rect = canvas.getBoundingClientRect();
    const left = rect.left, right = rect.right, top = rect.top, bottom = rect.bottom;
    const w = rect.width, h = rect.height;
    let minSx = Infinity, maxSx = -Infinity, minSy = Infinity, maxSy = -Infinity;
    const corners = [
        [worldMin.x, worldMin.y, worldMin.z], [worldMax.x, worldMin.y, worldMin.z],
        [worldMin.x, worldMax.y, worldMin.z], [worldMax.x, worldMax.y, worldMin.z],
        [worldMin.x, worldMin.y, worldMax.z], [worldMax.x, worldMin.y, worldMax.z],
        [worldMin.x, worldMax.y, worldMax.z], [worldMax.x, worldMax.y, worldMax.z],
    ];
    for (const [x, y, z] of corners) {
        _tempVec3.set(x, y, z).project(state.camera);
        if (_tempVec3.z > 1) continue; // behind camera: skip to avoid wrong/huge screen rect
        const sx = left + (_tempVec3.x * 0.5 + 0.5) * w;
        const sy = top + (-_tempVec3.y * 0.5 + 0.5) * h;
        if (sx < minSx) minSx = sx; if (sx > maxSx) maxSx = sx;
        if (sy < minSy) minSy = sy; if (sy > maxSy) maxSy = sy;
    }
    if (minSx === Infinity) return null;
    return {
        left: Math.max(left, minSx),
        right: Math.min(right, maxSx),
        top: Math.max(top, minSy),
        bottom: Math.min(bottom, maxSy),
    };
}

/** Fraction of screenAABB that overlaps lasso rect; 0 if no overlap or aabb is null/degenerate. */
function lassoOverlapRatio(lassoLeft, lassoRight, lassoTop, lassoBottom, aabb) {
    if (!aabb || aabb.right <= aabb.left || aabb.bottom <= aabb.top) return 0;
    const interLeft = Math.max(lassoLeft, aabb.left);
    const interRight = Math.min(lassoRight, aabb.right);
    const interTop = Math.max(lassoTop, aabb.top);
    const interBottom = Math.min(lassoBottom, aabb.bottom);
    if (interLeft >= interRight || interTop >= interBottom) return 0;
    const interArea = (interRight - interLeft) * (interBottom - interTop);
    const aabbArea = (aabb.right - aabb.left) * (aabb.bottom - aabb.top);
    return aabbArea > 0 ? interArea / aabbArea : 0;
}

const AABB_OVERLAP_THRESHOLD = 0.05;

function applyLassoSelection() {
    if (!lassoStart || !lassoEnd || !currentData) return;

    const x1 = lassoStart.x, y1 = lassoStart.y, x2 = lassoEnd.x, y2 = lassoEnd.y;
    const lassoLeft = Math.min(x1, x2);
    const lassoRight = Math.max(x1, x2);
    const lassoTop = Math.min(y1, y2);
    const lassoBottom = Math.max(y1, y2);

    // Coarse filter: expand lasso rect a bit so we include near-misses
    const PADDING = 30; // pixels around user rectangle for coarse AABB test
    const coarseLeft = lassoLeft - PADDING;
    const coarseRight = lassoRight + PADDING;
    const coarseTop = lassoTop - PADDING;
    const coarseBottom = lassoBottom + PADDING;

    const treeCandidates = [];
    const poleCandidates = [];
    const buildingCandidates = [];
    const vehicleCandidates = [];
    const conductorCandidates = [];

    // --- Coarse pass: screen-space AABB overlap with enlarged rect ---

    // Trees: AABB from X,Y,Z and crown_radius, height
    if (visibilityState.trees && currentData.trees && currentData.trees.length > 0) {
        currentData.trees.forEach(t => {
            const x = t.X ?? 0, y = t.Y ?? 0, z = t.Z ?? 0;
            const r = t.crown_radius ?? 1, H = t.height ?? 1;
            const worldMin = new THREE.Vector3(x - r, y - r, z);
            const worldMax = new THREE.Vector3(x + r, y + r, z + H);
            const aabb = getScreenAABB(worldMin, worldMax);
            if (lassoOverlapRatio(coarseLeft, coarseRight, coarseTop, coarseBottom, aabb) > AABB_OVERLAP_THRESHOLD) {
                treeCandidates.push(t);
            }
        });
    }

    // Poles, Buildings, Vehicles: world AABB from mesh
    if (visibilityState.poles) {
        (state.meshes.poles || []).forEach(mesh => {
            _lassoBox3.setFromObject(mesh);
            const aabb = getScreenAABB(_lassoBox3.min, _lassoBox3.max);
            if (lassoOverlapRatio(coarseLeft, coarseRight, coarseTop, coarseBottom, aabb) > AABB_OVERLAP_THRESHOLD) {
                poleCandidates.push(mesh);
            }
        });
    }
    if (visibilityState.buildings) {
        (state.meshes.buildings || []).forEach(mesh => {
            _lassoBox3.setFromObject(mesh);
            const aabb = getScreenAABB(_lassoBox3.min, _lassoBox3.max);
            if (lassoOverlapRatio(coarseLeft, coarseRight, coarseTop, coarseBottom, aabb) > AABB_OVERLAP_THRESHOLD) {
                buildingCandidates.push(mesh);
            }
        });
    }
    if (visibilityState.vehicles) {
        (state.meshes.vehicles || []).forEach(mesh => {
            _lassoBox3.setFromObject(mesh);
            const aabb = getScreenAABB(_lassoBox3.min, _lassoBox3.max);
            if (lassoOverlapRatio(coarseLeft, coarseRight, coarseTop, coarseBottom, aabb) > AABB_OVERLAP_THRESHOLD) {
                vehicleCandidates.push(mesh);
            }
        });
    }

    // Conductors: AABB from points
    if (visibilityState.conductors && currentData.conductors && currentData.conductors.length > 0) {
        currentData.conductors.forEach(c => {
            if (!c.points || c.points.length === 0) return;
            const min = new THREE.Vector3(Infinity, Infinity, Infinity);
            const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
            c.points.forEach(p => {
                min.x = Math.min(min.x, p[0]); min.y = Math.min(min.y, p[1]); min.z = Math.min(min.z, p[2]);
                max.x = Math.max(max.x, p[0]); max.y = Math.max(max.y, p[1]); max.z = Math.max(max.z, p[2]);
            });
            const aabb = getScreenAABB(min, max);
            if (lassoOverlapRatio(coarseLeft, coarseRight, coarseTop, coarseBottom, aabb) > AABB_OVERLAP_THRESHOLD) {
                conductorCandidates.push(c);
            }
        });
    }

    // --- Fine pass: vertex-based tests against the original lasso rect ---

    const canvas = state.renderer.domElement;
    const rect = canvas.getBoundingClientRect();
    const viewLeft = rect.left, viewTop = rect.top, viewW = rect.width, viewH = rect.height;

    function projectWorldToClient(x, y, z) {
        _tempVec3.set(x, y, z).project(state.camera);
        if (_tempVec3.z > 1) return null;
        const sx = viewLeft + (_tempVec3.x * 0.5 + 0.5) * viewW;
        const sy = viewTop + (-_tempVec3.y * 0.5 + 0.5) * viewH;
        return { sx, sy };
    }

    const pointInLasso = (sx, sy) =>
        sx >= lassoLeft && sx <= lassoRight && sy >= lassoTop && sy <= lassoBottom;

    const selected = [];

    // Trees: special case, reuse analytic box corners as \"vertices\"
    treeCandidates.forEach(t => {
        const x = t.X ?? 0, y = t.Y ?? 0, z = t.Z ?? 0;
        const r = t.crown_radius ?? 1, H = t.height ?? 1;
        const corners = [
            [x - r, y - r, z], [x + r, y - r, z],
            [x - r, y + r, z], [x + r, y + r, z],
            [x - r, y - r, z + H], [x + r, y - r, z + H],
            [x - r, y + r, z + H], [x + r, y + r, z + H],
        ];
        for (const [cx, cy, cz] of corners) {
            const p = projectWorldToClient(cx, cy, cz);
            if (!p) continue;
            if (pointInLasso(p.sx, p.sy)) {
                selected.push({ type: 'tree', id: t.id });
                break;
            }
        }
    });

    // Poles / Buildings / Vehicles: use geometry vertices
    function fineSelectFromMesh(mesh, type) {
        const geom = mesh.geometry;
        if (!geom || !geom.attributes || !geom.attributes.position) return;
        const pos = geom.attributes.position;
        const worldMatrix = mesh.matrixWorld;
        const count = pos.count;
        for (let i = 0; i < count; i++) {
            const vx = pos.getX(i);
            const vy = pos.getY(i);
            const vz = pos.getZ(i);
            _tempVec3.set(vx, vy, vz).applyMatrix4(worldMatrix);
            const p = projectWorldToClient(_tempVec3.x, _tempVec3.y, _tempVec3.z);
            if (!p) continue;
            if (pointInLasso(p.sx, p.sy)) {
                selected.push({ type, id: mesh.userData.id });
                return;
            }
        }
    }

    poleCandidates.forEach(mesh => fineSelectFromMesh(mesh, 'pole'));
    buildingCandidates.forEach(mesh => fineSelectFromMesh(mesh, 'building'));
    vehicleCandidates.forEach(mesh => fineSelectFromMesh(mesh, 'vehicle'));

    // Conductors: treat polyline points as vertices
    conductorCandidates.forEach(c => {
        for (const pWorld of c.points) {
            const p = projectWorldToClient(pWorld[0], pWorld[1], pWorld[2]);
            if (!p) continue;
            if (pointInLasso(p.sx, p.sy)) {
                selected.push({ type: 'conductor', id: c.id });
                break;
            }
        }
    });

    if (!selected.length) return;

    multiSelection = selected;
    lockedSelection = selected[selected.length - 1];
    updateHighlights();
    updateInfoPanel(lockedSelection);
    highlightItemInList(lockedSelection.type, lockedSelection.id);
    updateActionBar();
}

const TYPE_TO_API_KEY = { tree: 'trees', building: 'buildings', vehicle: 'vehicles', pole: 'poles', conductor: 'conductors' };

async function handleEdit(operation) {
    let targets = [];
    if (multiSelection.length > 0) {
        targets = multiSelection;
    } else if (lockedSelection) {
        targets = [lockedSelection];
    }
    if (targets.length === 0) return;

    const firstType = targets[0].type;
    const allSameType = targets.every(t => t.type === firstType);
    if (!allSameType) {
        alert("Select only one type of object for merge/split.");
        return;
    }
    const apiKey = TYPE_TO_API_KEY[firstType];
    if (!apiKey) {
        alert("Grouping is not supported for this object type.");
        return;
    }

    const ids = targets.map(t => t.id);
    const versionEl = document.getElementById('tile-version-id');
    const tileId = (versionEl && versionEl.value) || document.getElementById('tile-select').value;
    try {
        const result = await editGroup(tileId, apiKey, ids, operation);
        if (result[apiKey]) {
            // Update primitives for this type
            currentData[apiKey] = result[apiKey];
        }
        // Sync group relations if backend returned updated ones
        if (Array.isArray(result.group_relations)) {
            currentData.groupRelations = result.group_relations;
            buildGroupRelationsIndex(result.group_relations);
        }
        const groupsKeyMap = { buildings: 'buildingGroups', vehicles: 'vehicleGroups', trees: 'treeGroups' };
        const gk = groupsKeyMap[apiKey];
        if (gk && Array.isArray(result[gk.replace('Groups', '_groups')])) {
            currentData[gk] = result[gk.replace('Groups', '_groups')];
        }
        if (Array.isArray(result.building_groups)) currentData.buildingGroups = result.building_groups;
        if (Array.isArray(result.vehicle_groups)) currentData.vehicleGroups = result.vehicle_groups;
        if (Array.isArray(result.tree_groups)) currentData.treeGroups = result.tree_groups;

        // Cluster groups may have changed -- rebuild state
        state.macro.buildingGroups = currentData.buildingGroups || [];
        state.macro.vehicleGroups = currentData.vehicleGroups || [];
        state.macro.treeGroups = currentData.treeGroups || [];
        updateHighlights();
        updateInfoPanel(targets.length === 1 ? targets[0] : null);
        populateLists(currentData);
    } catch (e) {
        console.error(e);
        alert(`Edit failed: ${e.message}`);
    }
}

function onClick(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, state.camera);
    
    const objects = [];
    if (visibilityState.conductors) objects.push(...state.meshes.conductors);
    if (visibilityState.poles) objects.push(...state.meshes.poles);
    if (visibilityState.buildings) objects.push(...state.meshes.buildings);
    if (visibilityState.vehicles) objects.push(...state.meshes.vehicles);
    if (visibilityState.trees) objects.push(...state.meshes.trees);
    const hits = raycaster.intersectObjects(objects, false);
    
    if (hits.length > 0) {
        previewSelection = null; // so active = lockedSelection/multiSelection, not previous hover
        const hit = hits[0];
        const type = hit.object.userData.type;
        // Handle InstancedMesh for trees
        let id;
        if (type === 'tree' && hit.instanceId != null) {
             id = hit.object.userData.ids[hit.instanceId];
        } else {
             id = hit.object.userData.id;
        }
        
        if (event.shiftKey || event.ctrlKey) {
            // Multi-select
            const existingIdx = multiSelection.findIndex(item => item.type === type && item.id === id);
            if (existingIdx !== -1) {
                multiSelection.splice(existingIdx, 1);
            } else {
                multiSelection.push({ type, id });
            }
            lockedSelection = multiSelection.length > 0 ? multiSelection[multiSelection.length - 1] : null;
        } else {
            // Single select
            multiSelection = [{ type, id }];
            lockedSelection = { type, id };
        }
        
        updateHighlights();
        updateInfoPanel(lockedSelection);
        updateActionBar();
        highlightItemInList(lockedSelection ? lockedSelection.type : null, lockedSelection ? lockedSelection.id : null);
        
    }
}

function onDoubleClick(event) {
    if (event.button !== 0) return; // Only Left Double Click
    previewSelection = null;
    lockedSelection = null;
    multiSelection = [];
    updateHighlights();
    updateInfoPanel(null);
    updateActionBar();
}

function onPointerMove(event) {
    // Lasso handling: convert Shift+drag into active lasso after small movement
    if (lassoPending && lassoStart) {
        const dx = event.clientX - lassoStart.x;
        const dy = event.clientY - lassoStart.y;
        const dist2 = dx * dx + dy * dy;
        if (!lassoActive && dist2 > 25) { // ~5px threshold
            lassoActive = true;
            if (state.controls) {
                state.controls.enabled = false;
            }
        }
    }

    // If we're currently dragging a lasso, just update its rectangle
    if (lassoActive) {
        lassoEnd = { x: event.clientX, y: event.clientY };
        updateLassoVisual();
        return;
    }

    if (lockedSelection || multiSelection.length > 0) return;
    
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, state.camera);
    
    const objects = [];
    if (visibilityState.conductors) objects.push(...state.meshes.conductors);
    if (visibilityState.poles) objects.push(...state.meshes.poles);
    if (visibilityState.buildings) objects.push(...state.meshes.buildings);
    if (visibilityState.vehicles) objects.push(...state.meshes.vehicles);
    if (visibilityState.trees) objects.push(...state.meshes.trees);
    const hits = raycaster.intersectObjects(objects, false);
    
    if (hits.length > 0) {
        const hit = hits[0];
        const type = hit.object.userData.type;
        let id;
        if (type === 'tree' && hit.instanceId != null) {
             id = hit.object.userData.ids[hit.instanceId];
        } else {
             id = hit.object.userData.id;
        }
        previewSelection = { type, id };
        updateHighlights();
    } else {
        if (previewSelection) {
            previewSelection = null;
            updateHighlights();
        }
    }
}

function updateHighlights() {
    const active = previewSelection || lockedSelection || (multiSelection.length > 0 ? multiSelection[0] : null);

    // Build a set of all explicitly selected items (multi-select or single)
    const selectionSet = new Set();
    multiSelection.forEach(s => selectionSet.add(`${s.type}:${s.id}`));
    if (lockedSelection && !selectionSet.size) {
        selectionSet.add(`${lockedSelection.type}:${lockedSelection.id}`);
    }
    
    // Helper to check if an object is selected
    const isSelected = (type, id) =>
        selectionSet.has(`${type}:${id}`) ||
        (active && active.type === type && active.id === id);

    // Macro-instance highlighting: if a macro is active, treat its members as selected/grouped
    const macroMemberSets = {
        conductor: new Set(),
        building: new Set(),
        vehicle: new Set(),
        tree: new Set(),
    };
    if (active && active.type === 'macro' && currentData && Array.isArray(currentData.macro_instances)) {
        const macro = currentData.macro_instances.find(m => m.id === active.id && m.type === active.macroType);
        if (macro && Array.isArray(macro.member_ids)) {
            const mt = macro.member_type;
            if (macroMemberSets[mt]) {
                macro.member_ids.forEach(id => macroMemberSets[mt].add(id));
            }
        }
    }

    // Reset all to base color / dim if there is an active hover/selection
    state.meshes.conductors.forEach(m => {
        m.material.emissive.setHex(0);
        const raw = m.userData.originalColor;
        const defHex =
            (typeof raw === 'number')
                ? raw
                : (typeof raw === 'string' ? parseInt(raw.replace(/^#/, ''), 16) : 0x888888);
        let baseHex = defHex;
        if (currentColorScheme === 'instance') {
            baseHex = getInstanceColor('conductor', m.userData.id, defHex);
        } else if (currentColorScheme === 'macro') {
            baseHex = getMacroColor('conductor', m.userData.id, defHex);
        }
        if (!isNaN(baseHex)) m.material.color.setHex(baseHex);
        m.material.opacity = active ? OPACITY_DIM : 1.0;
        m.material.transparent = !!active;
    });
    
    state.meshes.poles.forEach(m => {
        const defaultEdge = m.userData.defaultEdgeColor ?? 0x000000;
        let baseBody = 0x000000;
        if (currentColorScheme === 'instance') {
            baseBody = getInstanceColor('pole', m.userData.id, 0x000000);
        } else if (currentColorScheme === 'macro') {
            baseBody = getMacroColor('pole', m.userData.id, 0x000000);
        }
        m.material.color.setHex(baseBody);
        m.material.opacity = active ? OPACITY_DIM : 0.4;
        m.children[0].material.color.setHex(defaultEdge);
    });

    // Reset buildings to default (and dim if anything is active)
    state.meshes.buildings.forEach(m => {
        const defColor = m.userData.defaultColor ?? 0x888888;
        const defOpacity = m.userData.defaultOpacity ?? 0.5;
        let baseColor = defColor;
        if (currentColorScheme === 'instance') {
            baseColor = getInstanceColor('building', m.userData.id, defColor);
        } else if (currentColorScheme === 'macro') {
            baseColor = getMacroColor('building', m.userData.id, defColor);
        }
        m.material.color.setHex(baseColor);
        m.material.opacity = active ? OPACITY_DIM : defOpacity;
    });

    // Reset vehicles to default (and dim if anything is active)
    state.meshes.vehicles.forEach(m => {
        const defColor = m.userData.defaultColor ?? 0x90ee90;
        const defOpacity = m.userData.defaultOpacity ?? 0.7;
        let baseColor = defColor;
        if (currentColorScheme === 'instance') {
            baseColor = getInstanceColor('vehicle', m.userData.id, defColor);
        } else if (currentColorScheme === 'macro') {
            baseColor = getMacroColor('vehicle', m.userData.id, defColor);
        }
        m.material.color.setHex(baseColor);
        m.material.opacity = active ? OPACITY_DIM : defOpacity;
    });

    // Pre-compute tree selection / group sets for mesh rebuild
    const selectedTreeIds = new Set();
    const groupedTreeIds = new Set();
    if (currentData && currentData.trees && currentData.trees.length > 0) {
        multiSelection.filter(s => s.type === 'tree').forEach(s => selectedTreeIds.add(s.id));
        if (lockedSelection && lockedSelection.type === 'tree') {
            selectedTreeIds.add(lockedSelection.id);
        }
        selectedTreeIds.forEach(id => {
            _getGroupedIds('tree', id).forEach(gid => groupedTreeIds.add(gid));
        });
    }

    // If nothing is active, just ensure trees are rendered in base mode and return
    if (!active) {
        rebuildTreeMeshes(selectedTreeIds, groupedTreeIds);
        return;
    }

    // Highlight selected poles (generic selection)
    state.meshes.poles.forEach(m => {
        if (isSelected('pole', m.userData.id)) {
            m.material.color.setHex(COLOR_SELECTED);
            m.material.opacity = 0.8;
        }
    });

    // Compute grouped ids for buildings and vehicles from current selection
    const groupedBuildingIds = new Set();
    const groupedVehicleIds = new Set();

    const selectedBuildings = multiSelection.filter(s => s.type === 'building');
    const selectedVehicles = multiSelection.filter(s => s.type === 'vehicle');

    if (!selectedBuildings.length && lockedSelection && lockedSelection.type === 'building') {
        selectedBuildings.push(lockedSelection);
    }
    if (!selectedVehicles.length && lockedSelection && lockedSelection.type === 'vehicle') {
        selectedVehicles.push(lockedSelection);
    }

    selectedBuildings.forEach(sel => {
        _getGroupedIds('building', sel.id).forEach(gid => groupedBuildingIds.add(gid));
    });

    selectedVehicles.forEach(sel => {
        _getGroupedIds('vehicle', sel.id).forEach(gid => groupedVehicleIds.add(gid));
    });

    // Highlight buildings: selected/macro in yellow, grouped in extension color
    state.meshes.buildings.forEach(m => {
        const bid = m.userData.id;
        if (isSelected('building', bid) || macroMemberSets.building.has(bid)) {
            m.material.color.setHex(COLOR_SELECTED);
            m.material.opacity = 0.9;
        } else if (groupedBuildingIds.has(bid)) {
            m.material.color.setHex(COLOR_EXTENSION);
            m.material.opacity = 0.8;
        }
    });

    // Highlight vehicles: selected/macro in yellow, grouped in extension color
    state.meshes.vehicles.forEach(m => {
        const vid = m.userData.id;
        if (isSelected('vehicle', vid) || macroMemberSets.vehicle.has(vid)) {
            m.material.color.setHex(COLOR_SELECTED);
            m.material.opacity = 0.9;
        } else if (groupedVehicleIds.has(vid)) {
            m.material.color.setHex(COLOR_EXTENSION);
            m.material.opacity = 0.8;
        }
    });

    // --- Tree highlighting via dedicated instanced meshes ---
    // Macro selection over trees should behave like per-tree group highlighting:
    const macroTreeMembers = macroMemberSets.tree;
    const allSelectedTreeIds = new Set(selectedTreeIds);
    macroTreeMembers.forEach(id => allSelectedTreeIds.add(id));
    rebuildTreeMeshes(allSelectedTreeIds, groupedTreeIds);

    // --- Relationship-based highlighting for active conductor (extensions, bifurcations, crosses, supports) ---
    if (active.type === 'conductor') {
        const id = String(active.id);
        const data = currentData;

        const extIds = (data.conductorExtensions && data.conductorExtensions[id])
            ? data.conductorExtensions[id].map(String)
            : [];
        const bifurcationIds = ((data.bifurcations && data.bifurcations[id]) ? data.bifurcations[id] : [])
            .filter(b => String(b) !== id);
        const crossIds = ((data.crosses && data.crosses[id]) ? data.crosses[id] : [])
            .filter(x => String(x) !== id);
        const supportPoleIds = (data.supportPoles && data.supportPoles[id]) ? data.supportPoles[id] : [];
        const supportBuildingIds = (data.supportBuildings && data.supportBuildings[id]) ? data.supportBuildings[id] : [];

        // Update Conductors
        state.meshes.conductors.forEach(m => {
            const mid = m.userData.id != null ? String(m.userData.id) : '';
            const isSelectedConductor = (mid === id);
            const isBifurcation = bifurcationIds.some(b => String(b) === mid);
            const isCross = crossIds.some(x => String(x) === mid);
            const isInExt = extIds.some(e => String(e) === mid);

            if (isSelectedConductor) {
                m.material.emissive.setHex(0x444400);
                m.material.color.setHex(COLOR_SELECTED);
                m.material.opacity = 1.0;
                m.material.transparent = false;
            } else if (isBifurcation) {
                m.material.color.setHex(COLOR_BIFURCATION);
                m.material.opacity = 1.0;
                m.material.transparent = false;
            } else if (isCross) {
                m.material.color.setHex(COLOR_CROSS);
                m.material.opacity = 1.0;
                m.material.transparent = false;
            } else if (isInExt) {
                m.material.color.setHex(COLOR_EXTENSION);
                m.material.opacity = 1.0;
                m.material.transparent = false;
            } else {
                m.material.opacity = OPACITY_DIM;
                m.material.transparent = true;
            }
        });

        // Update Poles: support = highlighted, uninvolved = dimmed
        state.meshes.poles.forEach(m => {
            if (supportPoleIds.includes(m.userData.id)) {
                m.material.color.setHex(COLOR_SUPPORT);
                m.material.opacity = 0.8;
                if (m.children[0] && m.children[0].material) {
                    m.children[0].material.color.setHex(COLOR_SUPPORT);
                }
            } else {
                m.material.opacity = OPACITY_DIM_POLE;
                const defaultEdge = m.userData.defaultEdgeColor ?? 0x000000;
                if (m.children[0] && m.children[0].material) {
                    m.children[0].material.color.setHex(defaultEdge);
                }
            }
        });

        // Update Buildings (Support only)
        state.meshes.buildings.forEach(m => {
            const bid = m.userData.id;
            if (supportBuildingIds.includes(bid)) {
                m.material.color.setHex(COLOR_SUPPORT);
                m.material.opacity = 0.8;
                if (m.children[0] && m.children[0].material) {
                    m.children[0].material.color.setHex(COLOR_SUPPORT);
                }
            }
        });
    } else if (active.type === 'building') {
        // Use selectionSet so all multi-selected buildings are yellow, not just active
        const grouped = groupedBuildingIds;
        state.meshes.buildings.forEach(m => {
            const bid = m.userData.id;
            if (isSelected('building', bid)) {
                m.material.color.setHex(COLOR_SELECTED);
                m.material.opacity = 0.9;
            } else if (grouped.has(bid)) {
                m.material.color.setHex(COLOR_EXTENSION);
                m.material.opacity = 0.8;
            }
        });
    } else if (active.type === 'vehicle') {
        const grouped = groupedVehicleIds;
        state.meshes.vehicles.forEach(m => {
            const vid = m.userData.id;
            if (isSelected('vehicle', vid)) {
                m.material.color.setHex(COLOR_SELECTED);
                m.material.opacity = 0.9;
            } else if (grouped.has(vid)) {
                m.material.color.setHex(COLOR_EXTENSION);
                m.material.opacity = 0.8;
            }
        });
    }
}

function updateInfoPanel(selection) {
    const el = document.getElementById('selectedInfo');
    // Derive an effective selection when caller passes null
    const sel =
        selection ||
        (multiSelection.length === 1 ? multiSelection[0] : null) ||
        lockedSelection;

    if (!sel && multiSelection.length === 0) {
        el.style.display = 'none';
        return;
    }
    el.style.display = 'block';
    
    if (multiSelection.length > 1) {
        // Basic multi-select header
        let html = `<b>Multi-Select</b> (${multiSelection.length} items)<br>` +
                   multiSelection.map(s => `${s.type} ${s.id}`).join(', ');

        // If selection is eligible for typed group relations, append summary lines
        const relSummary = buildRelationSummaryForSelection();
        if (relSummary) {
            html += `<br>${relSummary}`;
        }
        el.innerHTML = html;
        return;
    }

    const data = currentData;
    const relatedMacros = getMacrosForSelection(sel);

    // Base header: {class} {instance id}
    const baseLabel = (() => {
        if (!sel) return '';
        const t = sel.type;
        const id = sel.id;
        if (t === 'macro') {
            const macrosArr = Array.isArray(data.macro_instances) ? data.macro_instances : [];
            const m = relatedMacros[0] || macrosArr.find(mm => mm.id === id && mm.type === sel.macroType);
            const cls = m ? macroClassLabel(m) : sel.macroType || 'Macro';
            return `${cls} ${id}`;
        }
        const pretty = getSemanticClassName(t, (() => {
            const arr = data[t === 'tree' ? 'trees' : t === 'vehicle' ? 'vehicles' : t === 'building' ? 'buildings' : t === 'pole' ? 'poles' : ''];
            if (!arr) return null;
            const obj = arr.find(o => String(o.id) === String(id));
            return obj ? obj.sem_class : null;
        })());
        const idDisp = t === 'conductor' ? conductorDisplayLabel(id) : id;
        return `${pretty} ${idDisp}`;
    })();

    let html = `<b>${baseLabel}</b><br>`;

    // Macro lines: {macro class} {macro instance id}
    relatedMacros.forEach(m => {
        const clsLabel = macroClassLabel(m);
        html += `${clsLabel} ${m.id}<br>`;
    });

    // Conductor graph relations (same arrow style as typed cluster relations below)
    if (sel && sel.type === 'conductor') {
        const id = String(sel.id);
        const extList = (data.conductorExtensions && data.conductorExtensions[id]) ? data.conductorExtensions[id] : [];
        const bifList = ((data.bifurcations && data.bifurcations[id]) ? data.bifurcations[id] : [])
            .filter(u => String(u) !== id);
        const crossList = ((data.crosses && data.crosses[id]) ? data.crosses[id] : [])
            .filter(u => String(u) !== id);
        const parts = [];
        if (extList.length) {
            parts.push(`Extensions → ${extList.map(conductorDisplayLabel).join(', ')}`);
        }
        if (bifList.length) {
            parts.push(`Bifurcations → ${bifList.map(conductorDisplayLabel).join(', ')}`);
        }
        if (crossList.length) {
            parts.push(`Crosses → ${crossList.map(conductorDisplayLabel).join(', ')}`);
        }
        if (parts.length) html += parts.join('<br>') + '<br>';
    }

    if (sel && sel.type === 'pole') {
        const conductors = (data.poleToConductors && data.poleToConductors[sel.id]) || [];
        html += (conductors.length
            ? `Supported conductors ${conductors.map(conductorDisplayLabel).join(', ')}\n`
            : '\n');
    } else if (sel && sel.type === 'building') {
        const supportConductors =
            (data.buildingToSupportConductors && data.buildingToSupportConductors[sel.id]) || [];
        html += supportConductors.length
            ? `Supported conductors ${supportConductors.map(conductorDisplayLabel).join(', ')}\n`
            : '\n';
    }
    // Typed intra-group relations for single selection (buildings/vehicles/trees)
    const relDetail = buildRelationSummaryForSingle(sel);
    if (relDetail) {
        html += `<br>${relDetail}`;
    }

    el.innerHTML = html;
}

function buildRelationSummaryForSelection() {
    if (multiSelection.length < 2) return '';
    // All same type?
    const first = multiSelection[0];
    const allSameType = multiSelection.every(s => s.type === first.type);
    if (!allSameType) return '';
    const type = first.type;
    const memberTypeBySelection = {
        building: 'building',
        vehicle: 'vehicle',
        tree: 'tree',
    };
    const memberType = memberTypeBySelection[type];
    if (!memberType || !groupRelationsIndex[memberType]) return '';

    // All in same group?
    const collNameByType = {
        building: 'buildings',
        vehicle: 'vehicles',
        tree: 'trees',
    };
    const collName = collNameByType[type];
    const coll = currentData[collName] || [];
    const idToObj = new Map(coll.map(o => [o.id, o]));
    const gids = new Set();
    for (const sel of multiSelection) {
        const obj = idToObj.get(sel.id);
        if (!obj || obj.group_id == null) return '';
        gids.add(String(obj.group_id));
    }
    if (gids.size !== 1) return '';
    const gid = Array.from(gids)[0];

    const bucket = groupRelationsIndex[memberType].byGroup.get(gid);
    if (!bucket || !bucket.length) return '';

    const selSet = new Set(multiSelection.map(s => String(s.id)));
    const classCounts = {};
    bucket.forEach(r => {
        const a = String(r.a_id);
        const b = String(r.b_id);
        if (!selSet.has(a) || !selSet.has(b)) return;
        const cls = r.class || 'group';
        classCounts[cls] = (classCounts[cls] || 0) + 1;
    });

    const lines = [];
    RELATION_PRIORITY.forEach(cls => {
        const count = classCounts[cls];
        if (count > 0) {
            lines.push(`${cls.charAt(0).toUpperCase() + cls.slice(1)} (${count} pair${count > 1 ? 's' : ''})`);
        }
    });
    return lines.length ? lines.join('<br>') : '';
}

/** Resolve conductor row by scene id (string LAZ instance id). */
function conductorRowBySceneId(data, idOrKey) {
    const conds = data && data.conductors;
    if (!Array.isArray(conds)) return null;
    const s = String(idOrKey);
    return conds.find(c => String(c.id) === s) || null;
}

/** User-visible conductor label from graph relation id (numeric instance id). */
function conductorDisplayLabelFromData(data, idOrKey) {
    const row = conductorRowBySceneId(data, idOrKey);
    if (!row) return idOrKey != null ? String(idOrKey) : '';
    return String(row.id);
}

function conductorDisplayLabel(idOrKey) {
    return conductorDisplayLabelFromData(currentData, idOrKey);
}

function formatPeerRelationLabel(peerType, peerId) {
    const pt = (peerType || '').toLowerCase();
    const id = Number(peerId);
    const data = currentData || {};
    if (pt === 'building') {
        return `Building ${id}`;
    }
    if (pt === 'vehicle') {
        return `Vehicle ${id}`;
    }
    if (pt === 'pole') {
        return `Pole ${id}`;
    }
    if (pt === 'conductor') {
        return `Conductor ${conductorDisplayLabel(peerId)}`;
    }
    return `${peerType} ${id}`;
}

/** Typed label for same-class objects (building, vehicle, tree) in intra-group relations. */
function formatSameClassObjectLabel(objType, id) {
    const n = Number(id);
    if (objType === 'building') return `Building ${n}`;
    if (objType === 'vehicle') return `Vehicle ${n}`;
    if (objType === 'tree') return `Tree ${n}`;
    return String(id);
}

function buildRelationSummaryForSingle(selection) {
    if (!selection) return '';
    const rels = currentData.groupRelations || [];

    const mergedByClass = { adjacent: new Set(), near: new Set() };

    if (selection.type === 'tree') {
        const sid = String(selection.id);
        rels.forEach(r => {
            if (!r.peer_type || String(r.a_id) !== sid) return;
            const cls = (r.class || 'group').toLowerCase();
            if (!RELATION_PRIORITY.includes(cls)) return;
            mergedByClass[cls].add(formatPeerRelationLabel(r.peer_type, r.b_id));
        });
    } else {
        const pt = { building: 'building', vehicle: 'vehicle', pole: 'pole', conductor: 'conductor' }[selection.type];
        if (pt) {
            const pid = Number(selection.id);
            rels.forEach(r => {
                if (r.peer_type !== pt || Number(r.b_id) !== pid) return;
                const cls = (r.class || 'group').toLowerCase();
                if (!RELATION_PRIORITY.includes(cls)) return;
                mergedByClass[cls].add(`Tree ${r.a_id}`);
            });
        }
    }

    const memberTypeBySelection = {
        building: 'building',
        vehicle: 'vehicle',
        tree: 'tree',
    };
    const type = selection.type;
    const memberType = memberTypeBySelection[type];
    if (memberType && groupRelationsIndex[memberType]) {
        const collNameByType = {
            building: 'buildings',
            vehicle: 'vehicles',
            tree: 'trees',
        };
        const collName = collNameByType[type];
        const coll = currentData[collName] || [];
        const idToObj = new Map(coll.map(o => [o.id, o]));
        const obj = idToObj.get(selection.id);
        if (obj && obj.group_id != null) {
            const gid = String(obj.group_id);
            const bucket = groupRelationsIndex[memberType].byGroup.get(gid);
            const selfIdStr = String(selection.id);
            if (bucket && bucket.length) {
                bucket.forEach(r => {
                    if (r.peer_type) return;
                    const cls = (r.class || 'group').toLowerCase();
                    if (!RELATION_PRIORITY.includes(cls)) return;
                    const a = String(r.a_id);
                    const b = String(r.b_id);
                    if (a !== selfIdStr && b !== selfIdStr) return;
                    const other = (a === selfIdStr) ? b : a;
                    mergedByClass[cls].add(formatSameClassObjectLabel(type, other));
                });
            }
        }
    }

    const lines = [];
    RELATION_PRIORITY.forEach(cls => {
        const set = mergedByClass[cls];
        if (set && set.size) {
            const parts = Array.from(set).sort();
            lines.push(`${cls.charAt(0).toUpperCase() + cls.slice(1)} — ${parts.join(', ')}`);
        }
    });
    return lines.length ? lines.join('<br>') : '';
}

async function handleGroupRelationCycleShortcut() {
    if (multiSelection.length < 2) return;
    const first = multiSelection[0];
    const allSameType = multiSelection.every(s => s.type === first.type);
    if (!allSameType) return;

    const memberTypeBySelection = {
        building: 'building',
        vehicle: 'vehicle',
        tree: 'tree',
    };
    const memberType = memberTypeBySelection[first.type];
    if (!memberType || !groupRelationsIndex[memberType]) return;
    if (memberType === 'tree') return;

    const collNameByType = {
        building: 'buildings',
        vehicle: 'vehicles',
        tree: 'trees',
    };
    const collName = collNameByType[first.type];
    const coll = currentData[collName] || [];
    const idToObj = new Map(coll.map(o => [o.id, o]));

    const gids = new Set();
    for (const sel of multiSelection) {
        const obj = idToObj.get(sel.id);
        if (!obj || obj.group_id == null) return;
        gids.add(String(obj.group_id));
    }
    if (gids.size !== 1) return;
    const gid = Array.from(gids)[0];

    const bucket = groupRelationsIndex[memberType].byGroup.get(gid) || [];
    const selIds = multiSelection.map(s => String(s.id));
    const selSet = new Set(selIds);

    const present = new Set();
    bucket.forEach(r => {
        const a = String(r.a_id);
        const b = String(r.b_id);
        if (!selSet.has(a) || !selSet.has(b)) return;
        present.add(r.class || 'adjacent');
    });
    let currentCls = 'adjacent';
    for (const cls of RELATION_PRIORITY) {
        if (present.has(cls)) {
            currentCls = cls;
            break;
        }
    }
    const idx = RELATION_PRIORITY.indexOf(currentCls);
    const nextCls = RELATION_PRIORITY[(idx >= 0 ? idx + 1 : 0) % RELATION_PRIORITY.length];

    // Build batch updates: all unordered pairs in selection
    const updates = [];
    const pairKey = (a, b) => {
        const [aa, bb] = [String(a), String(b)].sort();
        return `${aa}-${bb}`;
    };
    const existingByPair = new Map();
    bucket.forEach(r => {
        const key = pairKey(r.a_id, r.b_id);
        existingByPair.set(key, r);
    });

    for (let i = 0; i < selIds.length; i++) {
        for (let j = i + 1; j < selIds.length; j++) {
            const a = selIds[i];
            const b = selIds[j];
            const key = pairKey(a, b);
            const existing = existingByPair.get(key);
            updates.push({
                id: existing ? existing.id : null,
                member_type: memberType,
                group_id: parseInt(gid, 10),
                a_id: parseInt(a, 10),
                b_id: parseInt(b, 10),
                cls: nextCls,
            });
        }
    }
    if (!updates.length) return;

    const tileIdEl = document.getElementById('tile-select');
    if (!tileIdEl || !tileIdEl.value) return;
    const tileId = tileIdEl.value;

    try {
        const resp = await fetch(`/api/group_relation/${encodeURIComponent(tileId)}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ relations: updates }),
        });
        if (!resp.ok) {
            console.error('Failed to update group relations', await resp.text());
            return;
        }
        const result = await resp.json();
        if (Array.isArray(result.group_relations)) {
            // Merge partial updates from the server into the full in‑memory list
            const existing = Array.isArray(currentData.groupRelations)
                ? currentData.groupRelations
                : [];
            const makeKey = r => {
                const pt = r.peer_type ? String(r.peer_type) : '';
                if (pt) {
                    return `${r.member_type}|${r.group_id}|${pt}|${r.a_id}-${r.b_id}`;
                }
                const [a, b] = [String(r.a_id), String(r.b_id)].sort();
                return `${r.member_type}|${r.group_id}|${a}-${b}`;
            };
            const mergedByKey = new Map();
            existing.forEach(r => {
                mergedByKey.set(makeKey(r), r);
            });
            result.group_relations.forEach(r => {
                mergedByKey.set(makeKey(r), r);
            });
            const merged = Array.from(mergedByKey.values());
            currentData.groupRelations = merged;
            buildGroupRelationsIndex(merged);
        }
        // Refresh info panel so new classes are visible
        const sel = multiSelection.length > 1 ? multiSelection[0] : lockedSelection;
        if (sel) updateInfoPanel(sel);
    } catch (err) {
        console.error('Error updating group relations', err);
    }
}
function updateActionBar() {
    const el = document.getElementById('action-bar');
    const hasAnySelection =
        multiSelection.length > 0 ||
        !!lockedSelection;

    if (hasAnySelection) {
        el.style.display = 'flex';
    } else {
        el.style.display = 'none';
    }
}

function openTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.dataset.tab === tabName) btn.classList.add('active');
    });

    // When switching to Macro tab, ensure a macro sub-tab is active
    if (tabName === 'macro') {
        if (!document.querySelector('.macro-tab-btn.active')) {
            openMacroSubTab('connector');
        }
    }
}

function openMacroSubTab(name) {
    document.querySelectorAll('.macro-tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.macro-tab-btn').forEach(btn => btn.classList.remove('active'));

    const content = document.getElementById('macro-tab-' + name);
    if (content) content.classList.add('active');

    document.querySelectorAll('.macro-tab-btn').forEach(btn => {
        if (btn.dataset.macroTab === name) btn.classList.add('active');
    });
}

function toggleColorScheme() {
    if (!currentData) return;
    if (currentColorScheme === 'semantic') {
        // Build deterministic instance colors on first switch
        if (
            !instanceColors.conductor.size &&
            !instanceColors.pole.size &&
            !instanceColors.building.size &&
            !instanceColors.vehicle.size
        ) {
            buildInstanceColorsFromData(currentData, false);
        }
        currentColorScheme = 'instance';
    } else if (currentColorScheme === 'instance') {
        // Ensure macro colors exist when entering macro mode
        if (!macroColors.size) {
            buildMacroColorsFromMacros(currentData.macro_instances || [], true);
        }
        currentColorScheme = 'macro';
    } else {
        currentColorScheme = 'semantic';
    }
}

function highlightItemInList(type, id) {
    // Clear all selections
    document.querySelectorAll('.obj-list li').forEach(el => el.classList.remove('selected'));

    if (!type || id === null || id === undefined) return;

    let tab = '';
    if (type === 'conductor') tab = 'extensions';
    else if (type === 'pole') tab = 'poles';
    else if (type === 'building') tab = 'buildings';
    else if (type === 'vehicle') tab = 'vehicles';
    else if (type === 'tree') tab = 'trees';

    if (tab === 'extensions') {
        // Find extension index for this conductor id
        if (currentData.extensionGroups) {
            const idx = currentData.extensionGroups.findIndex(g => g && g.includes(String(id)));
            if (idx !== -1) {
                const li = document.getElementById('comp-li-' + idx);
                if (li) {
                    li.classList.add('selected');
                    li.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    openTab('extensions');
                }
            }
        }
    } else {
        const li = document.getElementById(`${tab}-li-${id}`);
        if (li) {
            li.classList.add('selected');
            li.scrollIntoView({ behavior: 'smooth', block: 'center' });
            openTab(tab);
        }
    }
}

function populateLists(data) {
    // Extensions (from extensionGroups)
    const extensionList = document.getElementById('extensionList');
    extensionList.innerHTML = '';
    if (Array.isArray(data.extensionGroups)) {
        const sortedGroups = data.extensionGroups
            .map((group, index) => ({
                group,
                originalIndex: index,
                length: group.length,
            }))
            .sort((a, b) => b.length - a.length);

        sortedGroups.forEach(item => {
            const li = document.createElement('li');
            li.id = 'comp-li-' + item.originalIndex;
            const labels = (item.group || []).map(cid => conductorDisplayLabelFromData(data, cid));
            const joined = labels.join(', ');
            li.textContent = joined.length > 96 ? `${joined.slice(0, 93)}… (${item.length})` : joined;
            li.onclick = () => {
                if (item.group && item.group.length > 0) {
                    const firstId = item.group[0];
                    // Select first conductor of the component
                    multiSelection = [{ type: 'conductor', id: String(firstId) }];
                    lockedSelection = { type: 'conductor', id: String(firstId) };
                    updateHighlights();
                    updateInfoPanel(lockedSelection);
                    highlightItemInList('conductor', firstId);
                }
            };
            extensionList.appendChild(li);
        });
    }

    // Poles
    const poleList = document.getElementById('poleList');
    poleList.innerHTML = '';
    (data.poles || []).forEach(p => {
        const li = document.createElement('li');
        li.id = 'poles-li-' + p.id;
        li.textContent = `${getSemanticClassName('pole', p.sem_class)} ${p.id}`;
        li.onclick = () => {
            multiSelection = [{ type: 'pole', id: p.id }];
            lockedSelection = { type: 'pole', id: p.id };
            updateHighlights();
            updateInfoPanel(lockedSelection);
            highlightItemInList('pole', p.id);
        };
        poleList.appendChild(li);
    });

    // Buildings
    const buildingList = document.getElementById('buildingList');
    buildingList.innerHTML = '';
    (data.buildings || []).forEach(b => {
        const li = document.createElement('li');
        li.id = 'buildings-li-' + b.id;
        li.textContent = `${getSemanticClassName('building', b.sem_class)} ${b.id}`;
        li.onclick = () => {
            multiSelection = [{ type: 'building', id: b.id }];
            lockedSelection = { type: 'building', id: b.id };
            updateHighlights();
            updateInfoPanel(lockedSelection);
            highlightItemInList('building', b.id);
        };
        buildingList.appendChild(li);
    });

    // Vehicles
    const vehicleList = document.getElementById('vehicleList');
    vehicleList.innerHTML = '';
    (data.vehicles || []).forEach(v => {
        const li = document.createElement('li');
        li.id = 'vehicles-li-' + v.id;
        li.textContent = `${getSemanticClassName('vehicle', v.sem_class)} ${v.id}`;
        li.onclick = () => {
            multiSelection = [{ type: 'vehicle', id: v.id }];
            lockedSelection = { type: 'vehicle', id: v.id };
            updateHighlights();
            updateInfoPanel(lockedSelection);
            highlightItemInList('vehicle', v.id);
        };
        vehicleList.appendChild(li);
    });

    // Trees
    const treeList = document.getElementById('treeList');
    treeList.innerHTML = '';
    (data.trees || []).forEach(t => {
        const li = document.createElement('li');
        li.id = 'trees-li-' + t.id;
        li.textContent = `${getSemanticClassName('tree', t.sem_class)} ${t.id}`;
        li.onclick = () => {
            multiSelection = [{ type: 'tree', id: t.id }];
            lockedSelection = { type: 'tree', id: t.id };
            updateHighlights();
            updateInfoPanel(lockedSelection);
            highlightItemInList('tree', t.id);
        };
        treeList.appendChild(li);
    });

    const spanList = document.getElementById('macroSpanList');
    const gridList = document.getElementById('macroGridList');
    const bClusterList = document.getElementById('buildingClusterList');
    const vClusterList = document.getElementById('vehicleClusterList');
    const tClusterList = document.getElementById('treeClusterList');

    if (spanList) spanList.innerHTML = '';
    if (gridList) gridList.innerHTML = '';
    if (bClusterList) bClusterList.innerHTML = '';
    if (vClusterList) vClusterList.innerHTML = '';
    if (tClusterList) tClusterList.innerHTML = '';

    (data.connector_spans || []).forEach(s => {
        if (!spanList) return;
        const li = document.createElement('li');
        li.id = `macro-li-${s.id}`;
        li.textContent = `Span ${s.id}`;
        li.onclick = () => {
            lockedSelection = { type: 'macro', macroType: 'connector_span', id: s.id };
            multiSelection = [];
            updateHighlights();
            updateInfoPanel(lockedSelection);
        };
        spanList.appendChild(li);
    });
    (data.electrical_grids || []).forEach(g => {
        if (!gridList) return;
        const li = document.createElement('li');
        li.id = `macro-li-grid-${g.id}`;
        li.textContent = `Grid ${g.id}`;
        li.onclick = () => {
            lockedSelection = { type: 'macro', macroType: 'electrical_grid', id: g.id };
            multiSelection = [];
            updateHighlights();
            updateInfoPanel(lockedSelection);
        };
        gridList.appendChild(li);
    });

    function populateClusterList(groups, listEl, type) {
        if (!listEl || !Array.isArray(groups)) return;
        groups.forEach(g => {
            const li = document.createElement('li');
            li.textContent = `Cluster ${g.id} (${(g.members||[]).length} members)`;
            li.onclick = () => {
                multiSelection = (g.members||[]).map(mid => ({ type, id: mid }));
                lockedSelection = multiSelection.length ? multiSelection[0] : null;
                updateHighlights();
                updateInfoPanel(lockedSelection);
                updateActionBar();
            };
            listEl.appendChild(li);
        });
    }
    populateClusterList(data.buildingGroups || [], bClusterList, 'building');
    populateClusterList(data.vehicleGroups || [], vClusterList, 'vehicle');
    populateClusterList(data.treeGroups || [], tClusterList, 'tree');
}
