import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const state = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    meshes: {
        conductors: [],
        poles: [],
        buildings: [],
        vehicles: [],
        trees: [], // base tree mesh (instanced)
        treeMesh: null, // reference to the specific InstancedMesh
        treeHighlights: [], // overlay meshes for selected/grouped trees
    },
    macro: {
        connectorSpans: [],
        electricalGrids: [],
        buildingGroups: [],
        vehicleGroups: [],
        treeGroups: [],
        all: [],
    },
};

export function initScene(container) {
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0xffffff);

    state.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.01, 5000);
    // Start in a top-down view: camera along +Z looking towards the XY plane (z=0)
    state.camera.position.set(0, 0, 300);

    state.renderer = new THREE.WebGLRenderer({ antialias: true });
    state.renderer.setSize(window.innerWidth, window.innerHeight);
    container.appendChild(state.renderer.domElement);

    state.controls = new OrbitControls(state.camera, state.renderer.domElement);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.05;
    state.controls.screenSpacePanning = true;
    state.controls.minDistance = 0.1;
    state.controls.maxDistance = 5000;
    state.controls.listenToKeyEvents(window);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    state.scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(100, 100, 200);
    state.scene.add(dirLight);

    // Helpers
    const gridSize = 500;
    const gridDivisions = 50;
    const gridHelper = new THREE.GridHelper(gridSize, gridDivisions);
    gridHelper.rotation.x = Math.PI / 2;
    // Data lives in +X,+Y only; shift grid so it covers that quadrant (0..gridSize)
    gridHelper.position.set(gridSize / 2, gridSize / 2, 0);
    state.scene.add(gridHelper);

    const axesHelper = new THREE.AxesHelper(50);
    state.scene.add(axesHelper);

    window.addEventListener('resize', onWindowResize, false);
    
    animate();
}

function onWindowResize() {
    state.camera.aspect = window.innerWidth / window.innerHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    if (state.controls) state.controls.update();
    if (state.renderer && state.scene && state.camera) {
        state.renderer.render(state.scene, state.camera);
    }
}

export function clearScene() {
    // Remove existing meshes
    [
        ...state.meshes.conductors,
        ...state.meshes.poles,
        ...state.meshes.buildings,
        ...state.meshes.vehicles,
        ...state.meshes.treeHighlights,
    ].forEach(m => {
        state.scene.remove(m);
        if (m.geometry) m.geometry.dispose();
        if (m.material) {
            if (Array.isArray(m.material)) m.material.forEach(mat => mat.dispose());
            else m.material.dispose();
        }
    });
    if (state.meshes.treeMesh) {
        state.scene.remove(state.meshes.treeMesh);
        state.meshes.treeMesh.geometry.dispose();
        state.meshes.treeMesh.material.dispose();
    }
    
    // Clear arrays
    state.meshes.conductors = [];
    state.meshes.poles = [];
    state.meshes.buildings = [];
    state.meshes.vehicles = [];
    state.meshes.trees = [];
    state.meshes.treeMesh = null;
    state.meshes.treeHighlights = [];
}

export function renderGraph(data) {
    clearScene();

    // --- Render Poles ---
    if (data.poles) {
        data.poles.forEach(pole => {
            const min = new THREE.Vector3(...pole.min);
            const max = new THREE.Vector3(...pole.max);
            const size = new THREE.Vector3().subVectors(max, min);
            
            const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
            const material = new THREE.MeshBasicMaterial({ color: 0x000000, opacity: 0.4, transparent: true });
            const mesh = new THREE.Mesh(geometry, material);
            
            const edges = new THREE.EdgesGeometry(geometry);
            let edgeColor = 0x000000;
            if (pole.is_virtual) {
                edgeColor = pole.is_building_support ? 0x0000ff : 0xff0000;
            }
            const edgeMat = new THREE.LineBasicMaterial({ color: edgeColor });
            const wireframe = new THREE.LineSegments(edges, edgeMat);
            mesh.add(wireframe);

            mesh.userData.defaultEdgeColor = edgeColor;

            if (pole.is_virtual) {
                const center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);
                mesh.position.copy(center);
            } else {
                if (pole.rotation && pole.rotation.length >= 3) {
                    const R = pole.rotation;
                    const m = new THREE.Matrix4();
                    m.set(
                        R[0][0], R[1][0], R[2][0], 0,
                        R[0][1], R[1][1], R[2][1], 0,
                        R[0][2], R[1][2], R[2][2], 0,
                        0, 0, 0, 1
                    );
                    mesh.setRotationFromMatrix(m);
                }
                mesh.position.set(pole.position[0], pole.position[1], pole.position[2] - size.z / 2);
            }
            
            mesh.userData.type = 'pole';
            mesh.userData.id = pole.id;
            mesh.userData.info = pole;
            state.meshes.poles.push(mesh);
            state.scene.add(mesh);
        });
    }

    // --- Render Buildings ---
    if (data.buildings) {
        data.buildings.forEach(b => {
            let geometry;
            let rotation = null;
            
            if (b.hull && b.hull.vertices && b.hull.vertices.length > 0) {
                const vertices = b.hull.vertices.flat();
                const positions = new Float32Array(vertices);
                const indices = b.hull.faces.flat();
                geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                geometry.setIndex(indices);
                geometry.computeVertexNormals();
            } else {
                let center, size;
                if (b.obb) {
                    center = new THREE.Vector3(...b.obb.center);
                    size = new THREE.Vector3(...b.obb.size);
                    const R = b.obb.rotation;
                    rotation = new THREE.Matrix4();
                    rotation.set(
                        R[0][0], R[0][1], R[0][2], 0,
                        R[1][0], R[1][1], R[1][2], 0,
                        R[2][0], R[2][1], R[2][2], 0,
                        0, 0, 0, 1
                    );
                } else {
                    const min = new THREE.Vector3(...(b.min || [0,0,0]));
                    const max = new THREE.Vector3(...(b.max || [0,0,0]));
                    size = new THREE.Vector3().subVectors(max, min);
                    center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);
                }
                geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
            }

            const defaultColor = (data.buildingDefaultColors && data.buildingDefaultColors[b.id]) ?? 0x888888;
            const defaultOpacity = (data.buildingDefaultOpacity && data.buildingDefaultOpacity[b.id]) ?? 0.5;
            
            const material = new THREE.MeshBasicMaterial({ 
                color: defaultColor, 
                opacity: defaultOpacity, 
                transparent: true,
                side: THREE.DoubleSide
            });
            const mesh = new THREE.Mesh(geometry, material);
            
            mesh.userData.defaultColor = defaultColor;
            mesh.userData.defaultOpacity = defaultOpacity;
            mesh.userData.defaultEdgeColor = (defaultColor === 0x888888) ? 0x444444 : (defaultColor & 0xffffff);

            if (b.hull && b.hull.vertices) {
                // already in world coords
            } else if (b.obb) {
                mesh.position.set(...b.obb.center);
                mesh.setRotationFromMatrix(rotation);
            } else {
                const min = b.min || [0,0,0];
                const max = b.max || [0,0,0];
                mesh.position.set((min[0]+max[0])/2, (min[1]+max[1])/2, (min[2]+max[2])/2);
            }

            const edges = new THREE.EdgesGeometry(geometry);
            const edgeMat = new THREE.LineBasicMaterial({ 
                color: mesh.userData.defaultEdgeColor, 
                opacity: Math.min(0.7, defaultOpacity + 0.2), 
                transparent: true 
            });
            mesh.add(new THREE.LineSegments(edges, edgeMat));

            mesh.userData.type = 'building';
            mesh.userData.id = b.id;
            mesh.userData.info = b;
            state.meshes.buildings.push(mesh);
            state.scene.add(mesh);
        });
    }

    // --- Render Vehicles ---
    if (data.vehicles) {
        const colors = { 2: '#90EE90', 6: '#32CD32', 7: '#2E8B57', 8: '#006400' };
        data.vehicles.forEach(v => {
            let geometry;
            if (v.hull && v.hull.vertices) {
                const vertices = v.hull.vertices.flat();
                const positions = new Float32Array(vertices);
                const indices = v.hull.faces.flat();
                geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                geometry.setIndex(indices);
                geometry.computeVertexNormals();
            } else {
                const min = new THREE.Vector3(...(v.min || [0,0,0]));
                const max = new THREE.Vector3(...(v.max || [0,0,0]));
                const size = new THREE.Vector3().subVectors(max, min);
                geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
            }

            const hexStr = colors[v.sem_class] || '#90EE90';
            const defaultColor = typeof hexStr === 'number' ? hexStr : parseInt(hexStr.replace(/^#/, ''), 16);
            
            const material = new THREE.MeshBasicMaterial({ 
                color: defaultColor, 
                opacity: 0.7, 
                transparent: true, 
                side: THREE.DoubleSide 
            });
            const mesh = new THREE.Mesh(geometry, material);
            
            mesh.userData.defaultColor = defaultColor;
            mesh.userData.defaultOpacity = 0.7;
            
            if (!(v.hull && v.hull.vertices)) {
                const min = v.min || [0,0,0];
                const max = v.max || [0,0,0];
                mesh.position.set((min[0]+max[0])/2, (min[1]+max[1])/2, (min[2]+max[2])/2);
            }

            const edges = new THREE.EdgesGeometry(geometry);
            mesh.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: defaultColor, opacity: 0.7 })));

            mesh.userData.type = 'vehicle';
            mesh.userData.id = v.id;
            mesh.userData.info = v;
            state.meshes.vehicles.push(mesh);
            state.scene.add(mesh);
        });
    }

    // --- Render Trees (base green; selection/grouping overlay handled separately) ---
    if (data.trees && data.trees.length > 0) {
        const treeGeom = new THREE.ConeGeometry(1.0, 1.0, 10);
        treeGeom.rotateX(Math.PI / 2);
        const treeMat = new THREE.MeshBasicMaterial({
            color: 0x228b22,
            transparent: true,
            opacity: 0.8,
        });
        
        const count = data.trees.length;
        const instancedMesh = new THREE.InstancedMesh(treeGeom, treeMat, count);
        const dummy = new THREE.Object3D();
        const ids = [];

        data.trees.forEach((t, index) => {
            const x = t.X ?? 0;
            const y = t.Y ?? 0;
            const z0 = t.Z ?? 0;
            const h = t.height ?? 1.0;
            const r = t.crown_radius ?? 1.0;

            dummy.position.set(x, y, z0);
            dummy.scale.set(r, r, h);
            dummy.rotation.set(0, 0, 0);
            dummy.updateMatrix();

            instancedMesh.setMatrixAt(index, dummy.matrix);
            ids[index] = t.id;
        });

        instancedMesh.userData.type = 'tree';
        instancedMesh.userData.ids = ids; // Map instanceId -> treeId
        state.meshes.treeMesh = instancedMesh;
        state.meshes.trees.push(instancedMesh);
        state.scene.add(instancedMesh);
    }

    // --- Render Conductors ---
    const radius = data.conductorRadius ?? 0.1;
    if (data.conductors) {
        data.conductors.forEach((c, idx) => {
            if (c.points.length < 2) return;
            const points = c.points.map(p => new THREE.Vector3(...p));
            const curve = new THREE.CatmullRomCurve3(points);
            const tubeGeometry = new THREE.TubeGeometry(curve, 32, radius, 8);
            const material = new THREE.MeshPhongMaterial({ color: c.color });
            const mesh = new THREE.Mesh(tubeGeometry, material);
            
            mesh.userData = { 
                type: 'conductor', 
                id: c.id, 
                index: idx, 
                originalColor: c.color 
            };
            state.meshes.conductors.push(mesh);
            state.scene.add(mesh);
        });
    }
}
