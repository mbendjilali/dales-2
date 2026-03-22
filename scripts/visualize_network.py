import json
import argparse
import random
import numpy as np
import colorsys
import os
import glob
import sys
from typing import List, Dict, Any, Optional

import laspy

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.core.graph_edges import (
    conductor_pole_ids_from_edges,
    conductor_support_building_ids_from_edges,
)
from pipeline.lib.generate_json_graph import build_instance_graph
from pipeline.lib.find_supports import get_node_attachments, reconstruct_footprints


def generate_conductor_curve(conductor: Dict, steps: int = 20) -> List[List[float]]:
    model = conductor.get('model', {})
    curve2d = model.get('curve2d', {}).get('coeffs')
    plane = model.get('plane', {})
    origin = plane.get('origin')
    orientation = plane.get('orientation')
    
    start = conductor.get('startpoint')
    end = conductor.get('endpoint')

    if not (curve2d and origin and orientation and start and end):
        if start and end:
            return [start, end]
        return []

    O = np.array(origin)
    R = np.array(orientation)
    
    S_global = np.array(start)
    E_global = np.array(end)
    
    S_local = R.T @ (S_global - O)
    E_local = R.T @ (E_global - O)
    
    u_start = S_local[0]
    u_end = E_local[0]
    
    points = []
    u_values = np.linspace(u_start, u_end, steps)
    
    a, b, c = curve2d
    
    for u in u_values:
        v = a * u**2 + b * u + c
        P_local = np.array([u, v, 0])
        P_global = O + R @ P_local
        points.append(P_global.tolist())
        
    return points

def get_distinct_color(index: int = None, total: int = 50) -> str:
    hue = (index % total) / total
    # Keep saturation and lightness distinct
    sat = 0.8
    val = 0.9
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def _load_point_cloud(
    path: str,
    max_points: int = 200_000,
) -> List[Dict[str, Any]]:
    """
    Load a LAS/LAZ file and return a subsampled list of points
    with x, y, z and instance_id (if present).
    """
    las = laspy.read(path)
    xs = np.asarray(las.x, dtype=float)
    ys = np.asarray(las.y, dtype=float)
    zs = np.asarray(las.z, dtype=float)
    instance = getattr(las, "instance_id", None)

    n = xs.shape[0]
    if n == 0:
        return []
    step = max(1, n // max_points)

    points: List[Dict[str, Any]] = []
    for idx in range(0, n, step):
        rec: Dict[str, Any] = {
            "position": [float(xs[idx]), float(ys[idx]), float(zs[idx])],
        }
        if instance is not None:
            rec["instance_id"] = int(instance[idx])
        points.append(rec)
    return points


def visualize_network_web(
    json_file: str,
    output_file: Optional[str] = None,
    tolerance: float = float("inf"),
    pointcloud_file: Optional[str] = None,
) -> None:
    if output_file is None:
        base = os.path.basename(json_file)
        if base.lower().endswith(".json"):
            base = base[:-5]
        output_file = f"web_{base}.html"
    
    print(f"Loading {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if input is already a graph (has poles/conductors) or raw network (nodes/links)
    if 'poles' in data and 'conductors' in data:
        print("Input appears to be a pre-computed graph.")
        graph_data = data
        
        # Check for separated geometry file
        base_dir = os.path.dirname(json_file)
        filename = os.path.basename(json_file)
        if filename.startswith("graph_") and filename.endswith(".json"):
            tile_id = filename[6:-5]
            geom_filename = f"geom_{tile_id}.json"
            # Look in same dir or ../geometry/
            candidates = [
                os.path.join(base_dir, geom_filename),
                os.path.join(base_dir, "../geometry", geom_filename),
                os.path.join(base_dir, "geometry", geom_filename),
            ]
            geom_path = None
            for p in candidates:
                if os.path.exists(p):
                    geom_path = p
                    break
            
            if geom_path:
                print(f"Found geometry file: {geom_path}")
                with open(geom_path, 'r') as gf:
                    geom_data = json.load(gf)
                
                scale = float(geom_data.get("scale", 1.0))
                
                # Merge building hulls
                b_hulls = geom_data.get("buildingHulls", {})
                buildings = graph_data.get("buildings", [])
                merged_buildings = 0
                for b in buildings:
                    bid_str = str(b["id"])
                    if bid_str in b_hulls:
                        hull_data = b_hulls[bid_str]
                        # De-quantize vertices
                        q_verts = hull_data.get("vertices", [])
                        verts = [[float(v[0])/scale, float(v[1])/scale, float(v[2])/scale] for v in q_verts]
                        b["hull"] = {
                            "vertices": verts,
                            "faces": hull_data.get("faces", [])
                        }
                        merged_buildings += 1
                print(f"Merged geometry for {merged_buildings} buildings.")

                # Merge vehicle hulls
                v_hulls = geom_data.get("vehicleHulls", {})
                vehicles = graph_data.get("vehicles", [])
                merged_vehicles = 0
                for v in vehicles:
                    vid_str = str(v["id"])
                    if vid_str in v_hulls:
                        hull_data = v_hulls[vid_str]
                        # De-quantize vertices
                        q_verts = hull_data.get("vertices", [])
                        verts = [[float(v[0])/scale, float(v[1])/scale, float(v[2])/scale] for v in q_verts]
                        v["hull"] = {
                            "vertices": verts,
                            "faces": hull_data.get("faces", [])
                        }
                        merged_vehicles += 1
                print(f"Merged geometry for {merged_vehicles} vehicles.")
            else:
                print("No matching geometry file found (looked for geom_{tile_id}.json). Rendering bounding boxes only.")
    else:
        print("Input appears to be raw network data. Computing graph...")
        nodes = data.get('nodes', [])
        links = data.get('links', [])
        node_map = {n['id']: n for n in nodes}
        min_z = min(n.get('footprint', {}).get('min', [0, 0, float('inf')])[2] for n in nodes) if nodes else 0
        node_attachments, _ = get_node_attachments(nodes, links)
        nodes, rec_count = reconstruct_footprints(nodes, node_attachments, global_min_z=min_z)
        print(f"Reconstructed {rec_count} pole footprints.")

        # Compute the graph byproduct
        print("Computing extensions, bifurcations, and crosses...")
        # Update nodes in data before passing to build_instance_graph if needed?
        # build_instance_graph uses data['nodes']. 
        # We should update data['nodes'] with reconstructed ones.
        data['nodes'] = nodes
        graph_data, _geom_unused = build_instance_graph(data, tolerance=tolerance)

        # Save the graph byproduct
        base = os.path.basename(json_file)
        if base.lower().endswith(".json"):
            base = base[:-5]
        graph_output_file = f"graph_{base}.json"
        with open(graph_output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        print(f"Graph byproduct saved to {graph_output_file}")

    # Prepare Data for JSON Injection
    # Poles from graph_data already have cleaned fields
    # Some virtual poles may have no footprint; we synthesize small AABB footprints for them
    # using their centroids and a global ground Z so they are still visible in the viewer.
    js_poles = []
    # Estimate a global ground Z from existing pole footprints or Z-coordinates
    min_z_candidates: List[float] = []
    for pole in graph_data.get('poles', []):
        fp0 = pole.get('footprint')
        if fp0 and isinstance(fp0.get('min'), list) and len(fp0['min']) >= 3:
            try:
                min_z_candidates.append(float(fp0['min'][2]))
            except Exception:
                pass
        elif 'Z' in pole:
            try:
                min_z_candidates.append(float(pole['Z']))
            except Exception:
                pass
    global_min_z = min(min_z_candidates) if min_z_candidates else 0.0

    for pole in graph_data['poles']:
        fp = pole.get('footprint')
        # If pre-computed graph, footprint might be inside pole or pole IS the footprint dict wrapper.
        # When missing (common for synthetic virtual poles), derive a tiny box footprint around centroid.
        if not fp:
            x = pole.get('X')
            y = pole.get('Y')
            z = pole.get('Z')
            if x is None or y is None or z is None:
                continue
            half = 0.3  # ~0.6 m footprint in XY; adjust if needed
            fp = {
                'min': [x - half, y - half, global_min_z],
                'max': [x + half, y + half, z],
                'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                'is_virtual': True,
            }

        is_virtual = fp.get('is_virtual', False)
        
        # Check for building support status
        is_building_support = pole.get('is_building_support', False)
        
        pole_data = {
            'min': fp['min'],
            'max': fp['max'],
            'is_virtual': is_virtual,
            'is_building_support': is_building_support,
            'id': pole.get('id')
        }
        if not is_virtual:
            pole_data['position'] = [pole.get('X'), pole.get('Y'), pole.get('Z')]
            pole_data['rotation'] = fp.get('rotation')
        js_poles.append(pole_data)

    CONDUCTOR_RADIUS = 0.1

    pcl_points: List[Dict[str, Any]] = []
    if pointcloud_file:
        try:
            print(f"Loading point cloud from {pointcloud_file}...")
            pcl_points = _load_point_cloud(pointcloud_file)
            print(f"Loaded {len(pcl_points)} point cloud samples.")
        except Exception as e:
            print(f"Warning: failed to load point cloud '{pointcloud_file}': {e}")

    js_conductors = []
    bifurcation_map_js = {}
    crosses_map_js = {}
    
    comp_id_to_scene_ids: Dict[int, List[str]] = {}
    support_pole_map: Dict[str, List[int]] = {}
    support_building_map: Dict[str, List[int]] = {}
    proximity_map: Dict[str, List[Any]] = {}
    conductor_edge_id_to_scene_id: Dict[int, str] = {}

    for c in graph_data.get("conductors") or []:
        iid = c.get("instance_id")
        if iid is None:
            continue
        scene_id = str(int(iid))
        conductor_edge_id_to_scene_id[int(iid)] = scene_id

        points = generate_conductor_curve(c, steps=20)
        comp_idx = c.get("component", 0)
        color = get_distinct_color(
            index=comp_idx,
            total=max(
                len(set(x.get("component", 0) for x in (graph_data.get("conductors") or []))),
                1,
            ),
        )
        js_conductors.append(
            {
                "points": points,
                "color": color,
                "id": scene_id,
            }
        )
        comp_id_to_scene_ids.setdefault(comp_idx, []).append(scene_id)
        support_pole_map[scene_id] = conductor_pole_ids_from_edges(graph_data, int(iid))
        support_building_map[scene_id] = conductor_support_building_ids_from_edges(
            graph_data, int(iid)
        )

    def _add_cond_rel(m: Dict[str, List[str]], sa: str, sb: str) -> None:
        if sa == sb:
            return
        m.setdefault(sa, []).append(sb)
        m.setdefault(sb, []).append(sa)

    for e in graph_data.get("edges") or []:
        cls = str(e.get("class", "")).lower()
        if cls not in ("bifurcation", "cross"):
            continue
        at = str(e.get("a_type", "")).lower()
        bt = str(e.get("b_type", "")).lower()
        if at != "conductor" or bt != "conductor":
            continue
        try:
            ua, ub = int(e["a_id"]), int(e["b_id"])
        except (KeyError, TypeError, ValueError):
            continue
        sa, sb = conductor_edge_id_to_scene_id.get(ua), conductor_edge_id_to_scene_id.get(ub)
        if not sa or not sb:
            continue
        if cls == "bifurcation":
            _add_cond_rel(bifurcation_map_js, sa, sb)
        else:
            _add_cond_rel(crosses_map_js, sa, sb)

    extension_groups = list(comp_id_to_scene_ids.values())

    pole_to_conductors: Dict[int, List[str]] = {}
    for cond_id, pole_ids in support_pole_map.items():
        for pid in pole_ids:
            pole_to_conductors.setdefault(pid, []).append(cond_id)
    building_to_support_conductors: Dict[int, List[str]] = {}
    for cond_id, building_ids in support_building_map.items():
        for bid in building_ids:
            building_to_support_conductors.setdefault(bid, []).append(cond_id)
    building_to_proximity_conductors: Dict[int, List[str]] = {}
    for cond_id, building_ids in proximity_map.items():
        for bid in building_ids:
            building_to_proximity_conductors.setdefault(bid, []).append(cond_id)

    # Default colors for buildings: gray if uninvolved, distinct colors for support vs proximity
    BUILDING_COLOR_NONE = 0x888888
    BUILDING_COLOR_SUPPORT = 0x1565c0   # blue
    BUILDING_COLOR_PROXIMITY = 0xff9800  # orange
    building_default_colors = {}
    building_default_opacity = {}
    for b in graph_data.get('buildings', []):
        bid = b['id']
        is_support = bid in building_to_support_conductors
        is_proximity = bid in building_to_proximity_conductors
        if is_support:
            building_default_colors[bid] = BUILDING_COLOR_SUPPORT
            building_default_opacity[bid] = 0.6
        elif is_proximity:
            building_default_colors[bid] = BUILDING_COLOR_PROXIMITY
            building_default_opacity[bid] = 0.6
        else:
            building_default_colors[bid] = BUILDING_COLOR_NONE
            building_default_opacity[bid] = 0.3

    tree_prox_map: Dict[str, List[int]] = {}

    scene_data = {
        'poles': js_poles,
        'conductors': js_conductors,
        'conductorRadius': CONDUCTOR_RADIUS,
        'extensionGroups': extension_groups,
        'bifurcations': bifurcation_map_js,
        'crosses': crosses_map_js,
        'points': pcl_points,
        'buildings': graph_data.get('buildings', []),
        'vehicles': graph_data.get('vehicles', []),
        'trees': graph_data.get('trees', []),
        'buildingGroups': graph_data.get('building_groups', []),
        'vehicleGroups': graph_data.get('vehicle_groups', []),
        'vehicleDefaultColors': graph_data.get('vehicleDefaultColors', {}),
        'supportPoles': support_pole_map,
        'supportBuildings': support_building_map,
        'proximityBuildings': proximity_map,
        'poleToConductors': pole_to_conductors,
        'buildingToSupportConductors': building_to_support_conductors,
        'buildingToProximityConductors': building_to_proximity_conductors,
        'buildingDefaultColors': building_default_colors,
        'buildingDefaultOpacity': building_default_opacity,
        'treeProximity': tree_prox_map,
    }
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>3D Network Graph</title>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #f0f0f0; }}
        #info {{ position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.8); padding: 10px; font-family: sans-serif; pointer-events: none; }}
        #selectedInfo {{ position: absolute; bottom: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 8px; font-family: sans-serif; max-width: 320px; }}
        
        #sidebar {{
            display: flex;
            flex-direction: column;
            position: absolute;
            top: 50px;
            right: 10px;
            width: 300px;
            max-height: calc(100vh - 70px);
            background: white;
            border: 1px solid #ccc;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
            z-index: 1000;
            font-family: sans-serif;
        }}
        
        .tab-header {{
            display: flex;
            background: #eee;
            border-bottom: 1px solid #ccc;
        }}
        .tab-btn {{
            flex: 1;
            padding: 10px;
            border: none;
            background: none;
            cursor: pointer;
            font-weight: bold;
        }}
        .tab-btn.active {{
            background: white;
            border-bottom: 2px solid blue;
        }}
        
        .tab-content {{
            display: none;
            overflow-y: auto;
            flex: 1;
            padding: 10px;
        }}
        .tab-content.active {{
            display: block;
        }}
        
        ul.obj-list {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        ul.obj-list li {{
            padding: 8px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        ul.obj-list li.selected {{
            background-color: #e0e0ff;
            border-left: 4px solid blue;
        }}
        ul.obj-list li:hover {{
            background-color: #f0f0f0;
        }}
        .check-btn {{
            margin-left: 10px;
            cursor: pointer;
        }}
        
        #openListBtn {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 8px 16px;
            background: #fff;
            border: 1px solid #ccc;
            cursor: pointer;
            z-index: 100;
            font-family: sans-serif;
        }}
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }}
    </style>
    <!-- Import Map for OrbitControls -->
    <script type="importmap">
      {{
        "imports": {{
          "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
          "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
      }}
    </script>
</head>
<body>
    <div id="info">
                <h3>Network Graph 3D</h3>
                <p>Poles: Real (Black), Virtual (Red). Conductors: click to select.</p>
                <p><b>Single Click:</b> Select & Highlight | <b>Double Click:</b> Center Pivot on conductor | <b>Double Click background:</b> Reset View</p>
                <p>Left Drag: Rotate | Right Drag: Pan | Scroll: Zoom</p>
    </div>
    <div id="selectedInfo" style="display: none;"></div>
    
    <button id="openListBtn">List Objects</button>
    
    <div id="sidebar" style="display: none;">
        <div class="modal-header">
            <h3 style="margin: 0;">Inspector</h3>
            <button onclick="document.getElementById('sidebar').style.display='none'">Close</button>
        </div>
        <div class="tab-header">
            <button class="tab-btn active" onclick="openTab('components')">Components</button>
            <button class="tab-btn" onclick="openTab('poles')">Poles</button>
            <button class="tab-btn" onclick="openTab('buildings')">Buildings</button>
            <button class="tab-btn" onclick="openTab('vehicles')">Vehicles</button>
            <button class="tab-btn" onclick="openTab('trees')">Trees</button>
        </div>
        
        <div id="tab-components" class="tab-content active">
            <ul id="componentList" class="obj-list"></ul>
        </div>
        <div id="tab-poles" class="tab-content">
            <ul id="poleList" class="obj-list"></ul>
        </div>
        <div id="tab-buildings" class="tab-content">
            <ul id="buildingList" class="obj-list"></ul>
        </div>
        <div id="tab-vehicles" class="tab-content">
            <ul id="vehicleList" class="obj-list"></ul>
        </div>
        <div id="tab-trees" class="tab-content">
            <ul id="treeList" class="obj-list"></ul>
        </div>
    </div>

    <script>
        function openTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabName).classList.add('active');
            const btns = document.querySelectorAll('.tab-btn');
            if (tabName === 'components') btns[0].classList.add('active');
            if (tabName === 'poles') btns[1].classList.add('active');
            if (tabName === 'buildings') btns[2].classList.add('active');
            if (tabName === 'vehicles') btns[3].classList.add('active');
            if (tabName === 'trees') btns[4].classList.add('active');
        }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        // INJECTED DATA
        const data = {json.dumps(scene_data)};

        function rowForConductorId(tid) {{
            const s = String(tid);
            return (data.conductors || []).find(c => String(c.id) === s) || null;
        }}
        function conductorDisplayLabelHtml(tid) {{
            const row = rowForConductorId(tid);
            return row ? String(row.id) : String(tid);
        }}

        // Scene Setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xf0f0f0);
        // scene.fog = new THREE.Fog(0xf0f0f0, 20, 1000);

        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.01, 5000);
        camera.position.set(200, 200, 200);
        camera.up.set(0, 0, 1); // Z is up

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.screenSpacePanning = true;
        controls.minDistance = 0.1;
        controls.maxDistance = 5000;
        controls.listenToKeyEvents(window); 
        
        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
        dirLight.position.set(100, 100, 200);
        scene.add(dirLight);

        // Grid Helper
        const gridHelper = new THREE.GridHelper(1000, 50);
        gridHelper.rotation.x = Math.PI / 2;
        scene.add(gridHelper);
        
        const axesHelper = new THREE.AxesHelper(50);
        scene.add(axesHelper);

        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();

        // Store meshes for interaction
        const conductorMeshes = [];
        const poleMeshes = [];
        const buildingMeshes = [];
        const vehicleMeshes = [];
        const treeMeshes = [];
        let treeMesh = null;

        // --- Render Poles ---
        data.poles.forEach(pole => {{
            const min = new THREE.Vector3(...pole.min);
            const max = new THREE.Vector3(...pole.max);
            const size = new THREE.Vector3().subVectors(max, min);
            
            const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
            // Need Mesh for raycasting interaction
            const material = new THREE.MeshBasicMaterial({{ color: 0x000000, opacity: 0.2, transparent: true }});
            const mesh = new THREE.Mesh(geometry, material);
            
            // Edges for visual style (real=black, virtual=red, virtual building support=blue)
            const edges = new THREE.EdgesGeometry(geometry);
            let edgeColor = 0x000000;
            if (pole.is_virtual) {{
                if (pole.is_building_support) {{
                    edgeColor = 0x0000ff; // Blue for building supports
                }} else {{
                    edgeColor = 0xff0000; // Red for other virtual poles
                }}
            }}
            const edgeMat = new THREE.LineBasicMaterial({{ color: edgeColor }});
            const wireframe = new THREE.LineSegments(edges, edgeMat);
            mesh.add(wireframe);

            mesh.userData.defaultEdgeColor = edgeColor;

            if (pole.is_virtual) {{
                const center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);
                mesh.position.copy(center);
            }} else {{
                if (pole.rotation && pole.rotation.length >= 3) {{
                    const R = pole.rotation;
                    const m = new THREE.Matrix4();
                    m.set(
                        R[0][0], R[1][0], R[2][0], 0,
                        R[0][1], R[1][1], R[2][1], 0,
                        R[0][2], R[1][2], R[2][2], 0,
                        0, 0, 0, 1
                    );
                    mesh.setRotationFromMatrix(m);
                }}
                mesh.position.set(pole.position[0], pole.position[1], pole.position[2] - size.z / 2);
            }}
            
            mesh.userData.type = 'pole';
            mesh.userData.id = pole.id;
            mesh.userData.info = pole;
            poleMeshes.push(mesh);
            scene.add(mesh);
        }});

        // --- Render Buildings (Convex Hulls) ---
        if (data.buildings && data.buildings.length > 0) {{
            data.buildings.forEach(b => {{
                let geometry;
                if (b.hull && b.hull.vertices && b.hull.vertices.length > 0 && b.hull.faces && b.hull.faces.length > 0) {{
                    const vertices = b.hull.vertices.flat();
                    const positions = new Float32Array(vertices);
                    const indices = b.hull.faces.flat();
                    geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    geometry.setIndex(indices);
                    geometry.computeVertexNormals();
                }} else {{
                    // OBB or AABB box
                    let center, size, rotation;
                    if (b.obb) {{
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
                    }} else {{
                        const min = new THREE.Vector3(...(b.min || [0,0,0]));
                        const max = new THREE.Vector3(...(b.max || [0,0,0]));
                        size = new THREE.Vector3().subVectors(max, min);
                        center = new THREE.Vector3().addVectors(min, max).multiplyScalar(0.5);
                        rotation = new THREE.Matrix4();
                    }}
                    geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
                }}
                const defaultColor = (data.buildingDefaultColors && data.buildingDefaultColors[b.id]) ?? 0x888888;
                const defaultOpacity = (data.buildingDefaultOpacity && data.buildingDefaultOpacity[b.id]) ?? 0.3;
                const material = new THREE.MeshBasicMaterial({{ 
                    color: defaultColor, 
                    opacity: defaultOpacity, 
                    transparent: true,
                    side: THREE.DoubleSide
                }});
                const mesh = new THREE.Mesh(geometry, material);
                mesh.userData.defaultColor = defaultColor;
                mesh.userData.defaultOpacity = defaultOpacity;
                mesh.userData.defaultEdgeColor = (defaultColor === 0x888888) ? 0x444444 : (defaultColor & 0xffffff);
                if (b.hull && b.hull.vertices && b.hull.vertices.length > 0 && b.hull.faces && b.hull.faces.length > 0) {{
                    // Hull geometry is already in world coordinates; no transform
                }} else if (b.obb) {{
                    mesh.position.set(...b.obb.center);
                    mesh.setRotationFromMatrix(rotation);
                }} else {{
                    const min = b.min || [0,0,0];
                    const max = b.max || [0,0,0];
                    mesh.position.set(
                        (min[0] + max[0]) / 2,
                        (min[1] + max[1]) / 2,
                        (min[2] + max[2]) / 2
                    );
                    if (rotation) mesh.setRotationFromMatrix(rotation);
                }}
                const edges = new THREE.EdgesGeometry(geometry);
                const edgeMat = new THREE.LineBasicMaterial({{ color: mesh.userData.defaultEdgeColor, opacity: Math.min(0.7, defaultOpacity + 0.2), transparent: true }});
                const wireframe = new THREE.LineSegments(edges, edgeMat);
                mesh.add(wireframe);
                mesh.userData.type = 'building';
                mesh.userData.id = b.id;
                mesh.userData.info = b;
                buildingMeshes.push(mesh);
                scene.add(mesh);
            }});
        }}

        // --- Render Vehicles (cars/trucks from hull or box) ---
        if (data.vehicles && data.vehicles.length > 0) {{
            const defaultVehicleOpacity = 0.5;
            const colors = data.vehicleDefaultColors || {{ 3: '#90EE90', 4: '#2E8B57' }};
            data.vehicles.forEach(v => {{
                let geometry;
                if (v.hull && v.hull.vertices && v.hull.vertices.length > 0 && v.hull.faces && v.hull.faces.length > 0) {{
                    const vertices = v.hull.vertices.flat();
                    const positions = new Float32Array(vertices);
                    const indices = v.hull.faces.flat();
                    geometry = new THREE.BufferGeometry();
                    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    geometry.setIndex(indices);
                    geometry.computeVertexNormals();
                }} else {{
                    const min = new THREE.Vector3(...(v.min || [0,0,0]));
                    const max = new THREE.Vector3(...(v.max || [0,0,0]));
                    const size = new THREE.Vector3().subVectors(max, min);
                    geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
                }}
                const hexStr = colors[v.sem_class] || '#90EE90';
                const defaultColor = typeof hexStr === 'number' ? hexStr : parseInt(hexStr.replace(/^#/, ''), 16);
                const material = new THREE.MeshBasicMaterial({{ color: defaultColor, opacity: defaultVehicleOpacity, transparent: true, side: THREE.DoubleSide }});
                const mesh = new THREE.Mesh(geometry, material);
                mesh.userData.defaultColor = defaultColor;
                mesh.userData.defaultOpacity = defaultVehicleOpacity;
                mesh.userData.defaultEdgeColor = defaultColor;
                if (!(v.hull && v.hull.vertices && v.hull.vertices.length > 0)) {{
                    const min = v.min || [0,0,0];
                    const max = v.max || [0,0,0];
                    mesh.position.set((min[0]+max[0])/2, (min[1]+max[1])/2, (min[2]+max[2])/2);
                }}
                const edges = new THREE.EdgesGeometry(geometry);
                const edgeMat = new THREE.LineBasicMaterial({{ color: defaultColor, opacity: 0.7, transparent: true }});
                mesh.add(new THREE.LineSegments(edges, edgeMat));
                mesh.userData.type = 'vehicle';
                mesh.userData.id = v.id;
                mesh.userData.info = v;
                vehicleMeshes.push(mesh);
                scene.add(mesh);
            }});
        }}

        // --- Render Trees (instanced low-poly cones) ---
        if (data.trees && data.trees.length > 0) {{
            const treeGeom = new THREE.ConeGeometry(1.0, 1.0, 10); // base radius=1, height=1, low-poly
            treeGeom.rotateX(Math.PI / 2);
            const treeMat = new THREE.MeshBasicMaterial({{ color: 0x228b22, transparent: true, opacity: 0.8 }});
            const count = data.trees.length;
            treeMesh = new THREE.InstancedMesh(treeGeom, treeMat, count);
            const dummy = new THREE.Object3D();
            const ids = [];
            const baseColor = new THREE.Color(0x228b22);
            data.trees.forEach((t, index) => {{
                const x = t.X ?? 0;
                const y = t.Y ?? 0;
                const z0 = t.Z ?? 0;
                const h = t.height ?? 1.0;
                const r = t.crown_radius ?? 1.0;
                dummy.position.set(x, y, z0);
                // Scale cone: base radius -> crown_radius, height -> tree height
                const baseRadius = 1.0;
                const sx = r / baseRadius;
                const sy = r / baseRadius;
                const sz = h;
                dummy.scale.set(sx, sy, sz);
                dummy.rotation.set(0, 0, 0);
                dummy.updateMatrix();
                treeMesh.setMatrixAt(index, dummy.matrix);
                treeMesh.setColorAt(index, baseColor);
                ids[index] = t.id;
            }});
            if (treeMesh.instanceColor) {{
                treeMesh.instanceColor.needsUpdate = true;
            }}
            treeMesh.userData.type = 'tree';
            treeMesh.userData.ids = ids;
            scene.add(treeMesh);
            treeMeshes.push(treeMesh);
        }}

        // --- Render optional point cloud (PCL) ---
        if (Array.isArray(data.points) && data.points.length > 0) {{
            const positions = new Float32Array(data.points.length * 3);
            const colors = new Float32Array(data.points.length * 3);

            function colorFromInstance(id) {{
                const h = ((id * 2654435761) >>> 0) / 0xffffffff; 
                const s = 0.7;
                const l = 0.5;
                function hue2rgb(p, q, t) {{
                    if (t < 0) t += 1;
                    if (t > 1) t -= 1;
                    if (t < 1/6) return p + (q - p) * 6 * t;
                    if (t < 1/2) return q;
                    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                    return p;
                }}
                let r, g, b;
                if (s === 0) {{
                    r = g = b = l;
                }} else {{
                    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
                    const p = 2 * l - q;
                    r = hue2rgb(p, q, h + 1/3);
                    g = hue2rgb(p, q, h);
                    b = hue2rgb(p, q, h - 1/3);
                }}
                return [r, g, b];
            }}

            for (let i = 0; i < data.points.length; i++) {{
                const p = data.points[i].position;
                positions[3 * i + 0] = p[0];
                positions[3 * i + 1] = p[1];
                positions[3 * i + 2] = p[2];

                const inst = data.points[i].instance_id;
                const [r, g, b] = Number.isFinite(inst) ? colorFromInstance(inst) : [0.0, 0.0, 1.0];
                colors[3 * i + 0] = r;
                colors[3 * i + 1] = g;
                colors[3 * i + 2] = b;
            }}

            const geom = new THREE.BufferGeometry();
            geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            const mat = new THREE.PointsMaterial({{ size: 0.3, sizeAttenuation: true, vertexColors: true }});
            const cloud = new THREE.Points(geom, mat);
            scene.add(cloud);
        }}

        // --- Render Conductors (tubes) ---
        const radius = data.conductorRadius ?? 0.1;
        const tubularSegments = 32;
        const radialSegments = 8;
        
        data.conductors.forEach((c, idx) => {{
            const points = c.points.map(p => new THREE.Vector3(...p));
            const curve = new THREE.CatmullRomCurve3(points);
            const tubeGeometry = new THREE.TubeGeometry(curve, tubularSegments, radius, radialSegments);
            const material = new THREE.MeshPhongMaterial({{ color: c.color }});
            const mesh = new THREE.Mesh(tubeGeometry, material);
            mesh.userData = {{ type: 'conductor', id: c.id, index: idx, originalColor: c.color }};
            conductorMeshes.push(mesh);
            scene.add(mesh);
        }});

        // State
        let lockedSelection = null; // {{ type, id }}
        let previewSelection = null; // {{ type, id }}
        
        // Colors
        const COLOR_SELECTED = 0xffff00; // Yellow
        const COLOR_EXTENSION = 0xffa500; // Orange
        const COLOR_BIFURCATION = 0x800080; // Purple
        const COLOR_CROSS = 0xff0000; // Red
        const COLOR_PROXIMITY = 0xff9800; // Orange
        const COLOR_SUPPORT = 0x1565c0; // Blue
        const OPACITY_DIM = 0.2;
        const OPACITY_DIM_POLE = 0.08;  // Uninvolved poles when hovering conductor

        function updateHighlights() {{
            const active = previewSelection || lockedSelection;
            
            // Reset all to default
            conductorMeshes.forEach(m => {{
                m.material.emissive.setHex(0);
                const raw = m.userData.originalColor;
                const hex = (typeof raw === 'number') ? raw : (typeof raw === 'string' ? parseInt(raw.replace(/^#/, ''), 16) : 0x888888);
                if (!isNaN(hex)) m.material.color.setHex(hex);
                m.material.opacity = active ? OPACITY_DIM : 1.0;
                m.material.transparent = !!active;
            }});
            poleMeshes.forEach(m => {{
                const defaultEdge = m.userData.defaultEdgeColor ?? 0x000000;
                m.material.color.setHex(0x000000);
                m.material.opacity = active ? OPACITY_DIM : 0.2;
                m.children[0].material.color.setHex(defaultEdge);
            }});
            buildingMeshes.forEach(m => {{
                const defColor = m.userData.defaultColor ?? 0x888888;
                const defOpacity = m.userData.defaultOpacity ?? 0.3;
                const defEdge = m.userData.defaultEdgeColor ?? 0x444444;
                m.material.color.setHex(defColor);
                m.material.opacity = active ? OPACITY_DIM : defOpacity;
                m.children[0].material.color.setHex(defEdge);
            }});
            vehicleMeshes.forEach(m => {{
                const defColor = m.userData.defaultColor ?? 0x90ee90;
                const defOpacity = m.userData.defaultOpacity ?? 0.5;
                const defEdge = m.userData.defaultEdgeColor ?? defColor;
                m.material.color.setHex(defColor);
                m.material.opacity = active ? OPACITY_DIM : defOpacity;
                if (m.children[0] && m.children[0].material) {{
                    m.children[0].material.color.setHex(defEdge);
                }}
            }});
            if (treeMesh) {{
                const baseColor = new THREE.Color(0x228b22);
                for (let i = 0; i < treeMesh.count; i++) {{
                    treeMesh.setColorAt(i, baseColor);
                }}
                if (treeMesh.instanceColor) {{
                    treeMesh.instanceColor.needsUpdate = true;
                }}
                treeMesh.material.opacity = 0.8;
            }}

            if (!active) return;

            // Highlight based on selection
            if (active.type === 'conductor') {{
                const id = String(active.id);
                const extGroup = data.extensionGroups ? data.extensionGroups.find(g => g && g.includes(id)) : null;
                const extIds = (extGroup && extGroup.length) ? extGroup : [id];
                const bifurcationIds = (data.bifurcations && data.bifurcations[id]) ? data.bifurcations[id] : [];
                const crossIds = (data.crosses && data.crosses[id]) ? data.crosses[id] : [];
                const supportPoleIds = (data.supportPoles && data.supportPoles[id]) ? data.supportPoles[id] : [];
                const supportBuildingIds = (data.supportBuildings && data.supportBuildings[id]) ? data.supportBuildings[id] : [];
                const proximityIds = (data.proximityBuildings && data.proximityBuildings[id]) ? data.proximityBuildings[id] : [];

                // Update Conductors: set all to dimmed first, then highlight involved (consistent string comparison)
                conductorMeshes.forEach(m => {{
                    const mid = m.userData.id != null ? String(m.userData.id) : '';
                    const isSelected = (mid === id);
                    const isBifurcation = bifurcationIds.some(b => String(b) === mid);
                    const isCross = crossIds.some(x => String(x) === mid);
                    const isInExt = extIds.some(e => String(e) === mid);
                    if (isSelected) {{
                        m.material.emissive.setHex(0x444400);
                        m.material.color.setHex(COLOR_SELECTED);
                        m.material.opacity = 1.0;
                        m.material.transparent = false;
                    }} else if (isBifurcation) {{
                        m.material.color.setHex(COLOR_BIFURCATION);
                        m.material.opacity = 1.0;
                        m.material.transparent = false;
                    }} else if (isCross) {{
                        m.material.color.setHex(COLOR_CROSS);
                        m.material.opacity = 1.0;
                        m.material.transparent = false;
                    }} else if (isInExt) {{
                        m.material.color.setHex(COLOR_EXTENSION);
                        m.material.opacity = 1.0;
                        m.material.transparent = false;
                    }} else {{
                        m.material.opacity = OPACITY_DIM;
                        m.material.transparent = true;
                    }}
                }});

                // Update Poles: support = highlighted, uninvolved = dimmed
                poleMeshes.forEach(m => {{
                    if (supportPoleIds.includes(m.userData.id)) {{
                        m.material.color.setHex(COLOR_SUPPORT);
                        m.material.opacity = 0.8;
                        m.children[0].material.color.setHex(COLOR_SUPPORT);
                    }} else {{
                        m.material.opacity = OPACITY_DIM_POLE;
                        m.children[0].material.color.setHex(m.userData.defaultEdgeColor ?? 0x000000);
                    }}
                }});

                // Update Buildings (Support & Proximity)
                buildingMeshes.forEach(m => {{
                    if (supportBuildingIds.includes(m.userData.id)) {{
                        m.material.color.setHex(COLOR_SUPPORT);
                        m.material.opacity = 0.8;
                        m.children[0].material.color.setHex(COLOR_SUPPORT);
                    }} else if (proximityIds.includes(m.userData.id)) {{
                        m.material.color.setHex(COLOR_PROXIMITY);
                        m.material.opacity = 0.8;
                        m.children[0].material.color.setHex(COLOR_PROXIMITY);
                    }}
                }});

            }} else if (active.type === 'pole') {{
                const id = active.id;
                const m = poleMeshes.find(m => m.userData.id === id);
                if (m) {{
                    m.material.color.setHex(COLOR_SELECTED);
                    m.material.opacity = 0.8;
                    m.children[0].material.color.setHex(COLOR_SELECTED);
                }}
            }} else if (active.type === 'building') {{
                const id = active.id;
                const bInfo = (data.buildings || []).find(b => b.id === id) || null;
                const bgid = bInfo && bInfo.group_id != null ? bInfo.group_id : null;
                const bg = bgid != null ? (data.buildingGroups || []).find(g => g.id === bgid) : null;
                const peers = bg && Array.isArray(bg.members) ? bg.members.filter(m => m !== id) : [];
                buildingMeshes.forEach(m => {{
                    if (m.userData.id === id) {{
                        m.material.color.setHex(COLOR_SELECTED);
                        m.material.opacity = 0.9;
                        m.children[0].material.color.setHex(COLOR_SELECTED);
                    }} else if (peers.includes(m.userData.id)) {{
                        m.material.color.setHex(COLOR_EXTENSION);
                        m.material.opacity = 0.8;
                        m.children[0].material.color.setHex(COLOR_EXTENSION);
                    }}
                }});
            }} else if (active.type === 'vehicle') {{
                const id = active.id;
                const vInfo = (data.vehicles || []).find(v => v.id === id) || null;
                const vgid = vInfo && vInfo.group_id != null ? vInfo.group_id : null;
                const vg = vgid != null ? (data.vehicleGroups || []).find(g => g.id === vgid) : null;
                const peers = vg && Array.isArray(vg.members) ? vg.members.filter(m => m !== id) : [];
                vehicleMeshes.forEach(m => {{
                    if (m.userData.id === id) {{
                        m.material.color.setHex(COLOR_SELECTED);
                        m.material.opacity = 0.9;
                        m.children[0].material.color.setHex(COLOR_SELECTED);
                    }} else if (peers.includes(m.userData.id)) {{
                        m.material.color.setHex(COLOR_EXTENSION);
                        m.material.opacity = 0.8;
                        m.children[0].material.color.setHex(COLOR_EXTENSION);
                    }}
                }});
            }} else if (active.type === 'tree') {{
                if (treeMesh) {{
                    const ids = treeMesh.userData.ids || [];
                    const baseColor = new THREE.Color(0x228b22);
                    const highlightColor = new THREE.Color(COLOR_SELECTED);
                    const id = active.id;
                    ids.forEach((tid, index) => {{
                        treeMesh.setColorAt(index, tid === id ? highlightColor : baseColor);
                    }});
                    if (treeMesh.instanceColor) {{
                        treeMesh.instanceColor.needsUpdate = true;
                    }}
                    treeMesh.material.opacity = 0.9;
                }}
            }}
        }}

        // Update Info Panel
        function updateInfoPanel(selection) {{
            const el = document.getElementById('selectedInfo');
            if (!selection) {{
                el.style.display = 'none';
                return;
            }}
            el.style.display = 'block';
            
            if (selection.type === 'conductor') {{
                const id = selection.id;
                const extGroup = data.extensionGroups.find(g => g.includes(id)) || [id];
                const bifList = data.bifurcations[id] || [];
                const crossList = data.crosses[id] || [];
                const supPoles = data.supportPoles[id] || [];
                const supBuilds = data.supportBuildings[id] || [];
                const proxBuilds = data.proximityBuildings[id] || [];
                
                const disp = conductorDisplayLabelHtml(id);
                el.innerHTML =
                    '<b>Conductor: ' + disp + '</b><br>' +
                    'Component Size: ' + extGroup.length + '<br>' +
                    'Bifurcations: ' + (bifList.length ? bifList.map(conductorDisplayLabelHtml).join(', ') : 'none') + '<br>' +
                    'Crosses: ' + (crossList.length ? crossList.map(conductorDisplayLabelHtml).join(', ') : 'none') + '<br>' +
                    'Support Poles: ' + supPoles.join(', ') + '<br>' +
                    'Support Buildings: ' + supBuilds.join(', ') + '<br>' +
                    'Proximity Buildings: ' + proxBuilds.join(', ');
            }} else if (selection.type === 'pole') {{
                const conductors = data.poleToConductors && data.poleToConductors[selection.id] || [];
                el.innerHTML =
                    '<b>Pole: ' + selection.id + '</b><br>' +
                    'Connected conductors: ' + (conductors.length ? conductors.map(conductorDisplayLabelHtml).join(', ') : 'none');
            }} else if (selection.type === 'building') {{
                const supportConductors = data.buildingToSupportConductors && data.buildingToSupportConductors[selection.id] || [];
                const proximityConductors = data.buildingToProximityConductors && data.buildingToProximityConductors[selection.id] || [];
                const bInfo = (data.buildings || []).find(b => b.id === selection.id) || null;
                const bgid = bInfo && bInfo.group_id != null ? bInfo.group_id : null;
                const bg = bgid != null ? (data.buildingGroups || []).find(g => g.id === bgid) : null;
                const members = bg && Array.isArray(bg.members) ? bg.members.filter(m => m !== selection.id) : [];
                el.innerHTML =
                    '<b>Building: ' + selection.id + '</b><br>' +
                    'Group id: ' + (bgid != null ? bgid : 'none') + '<br>' +
                    'Group peers: ' + (members.length ? members.join(', ') : 'none') + '<br>' +
                    'Support conductors: ' + (supportConductors.length ? supportConductors.map(conductorDisplayLabelHtml).join(', ') : 'none') + '<br>' +
                    'Proximity conductors: ' + (proximityConductors.length ? proximityConductors.map(conductorDisplayLabelHtml).join(', ') : 'none');
            }} else if (selection.type === 'vehicle') {{
                const vInfo = (data.vehicles || []).find(v => v.id === selection.id) || null;
                const vgid = vInfo && vInfo.group_id != null ? vInfo.group_id : null;
                const vg = vgid != null ? (data.vehicleGroups || []).find(g => g.id === vgid) : null;
                const members = vg && Array.isArray(vg.members) ? vg.members.filter(m => m !== selection.id) : [];
                el.innerHTML =
                    '<b>Vehicle: ' + selection.id + '</b><br>' +
                    'Class: ' + (vInfo ? vInfo.sem_class : 'n/a') + '<br>' +
                    'Group id: ' + (vgid != null ? vgid : 'none') + '<br>' +
                    'Group peers: ' + (members.length ? members.join(', ') : 'none');
            }} else if (selection.type === 'tree') {{
                const tInfo = (data.trees || []).find(t => t.id === selection.id) || null;
                let nearConductors = [];
                if (data.treeProximity) {{
                    for (const [cid, tids] of Object.entries(data.treeProximity)) {{
                        if (tids.includes(selection.id)) nearConductors.push(cid);
                    }}
                }}
                el.innerHTML =
                    '<b>Tree: ' + selection.id + '</b><br>' +
                    'Height: ' + (tInfo ? tInfo.height.toFixed(2) : 'n/a') + ' m<br>' +
                    'Crown radius: ' + (tInfo ? tInfo.crown_radius.toFixed(2) : 'n/a') + ' m<br>' +
                    'Near conductors: ' + (nearConductors.length ? nearConductors.map(conductorDisplayLabelHtml).join(', ') : 'none');
            }}
        }}

        // --- Interaction ---
        
        function highlightItemInList(type, id) {{
            // De-select all
            document.querySelectorAll('.obj-list li').forEach(el => el.classList.remove('selected'));
            
            if (!type || !id) return;
            
            // Map types to tab names
            let tab = '';
            if (type === 'conductor') tab = 'components';
            else if (type === 'pole') tab = 'poles';
            else if (type === 'building') tab = 'buildings';
            else if (type === 'vehicle') tab = 'vehicles';
            else if (type === 'tree') tab = 'trees';
            
            if (tab === 'components') {{
                // Find component index for conductor id
                const idx = data.extensionGroups.findIndex(g => g.includes(id));
                if (idx !== -1) {{
                    // Our list is sorted, need to find the LI with data-id=originalIndex
                    const li = document.getElementById('comp-li-' + idx);
                    if (li) {{
                        li.classList.add('selected');
                        li.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        openTab('components');
                    }}
                }}
            }} else {{
                const li = document.getElementById(tab + '-li-' + id);
                if (li) {{
                    li.classList.add('selected');
                    li.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    openTab(tab);
                }}
            }}
        }}

        // Lists Population
        // Components (Sorted)
        const sortedGroups = data.extensionGroups.map((group, index) => ({{
            group: group,
            originalIndex: index,
            length: group.length
        }})).sort((a, b) => b.length - a.length);
        
        const componentList = document.getElementById('componentList');
        sortedGroups.forEach(item => {{
            const li = document.createElement('li');
            li.id = 'comp-li-' + item.originalIndex;
            const compLabels = item.group.map(conductorDisplayLabelHtml).join(', ');
            const compText = compLabels.length > 80 ? compLabels.slice(0, 77) + '…' : compLabels;
            li.innerHTML = `<span>${{compText}} (${{item.length}})</span>`;
            
            const chk = document.createElement('input');
            chk.type = 'checkbox';
            chk.className = 'check-btn';
            chk.onclick = (e) => e.stopPropagation();
            li.appendChild(chk);
            
            li.onclick = () => {{
                // Select first conductor of component
                const id = item.group[0];
                lockSelection('conductor', id);
            }};
            componentList.appendChild(li);
        }});

        // Poles
        const poleList = document.getElementById('poleList');
        data.poles.forEach(p => {{
            const li = document.createElement('li');
            li.id = 'poles-li-' + p.id;
            li.textContent = `Pole ${{p.id}}`;
            li.onclick = () => lockSelection('pole', p.id);
            poleList.appendChild(li);
        }});

        // Buildings
        const buildingList = document.getElementById('buildingList');
        data.buildings.forEach(b => {{
            const li = document.createElement('li');
            li.id = 'buildings-li-' + b.id;
            li.textContent = `Building ${{b.id}}`;
            li.onclick = () => lockSelection('building', b.id);
            buildingList.appendChild(li);
        }});

        // Vehicles
        const vehicleList = document.getElementById('vehicleList');
        (data.vehicles || []).forEach(v => {{
            const li = document.createElement('li');
            li.id = 'vehicles-li-' + v.id;
            li.textContent = `Vehicle ${{v.id}} (class ${{v.sem_class}})`;
            li.onclick = () => lockSelection('vehicle', v.id);
            vehicleList.appendChild(li);
        }});

        // Trees
        const treeList = document.getElementById('treeList');
        (data.trees || []).forEach(t => {{
            const li = document.createElement('li');
            li.id = 'trees-li-' + t.id;
            li.textContent = `Tree ${{t.id}}`;
            li.onclick = () => lockSelection('tree', t.id);
            treeList.appendChild(li);
        }});

        function lockSelection(type, id) {{
            lockedSelection = {{ type, id }};
            previewSelection = null;
            updateHighlights();
            updateInfoPanel(lockedSelection);
            highlightItemInList(type, id);
            
            // Move camera
            moveCameraToSelection(type, id);
        }}

        function moveCameraToSelection(type, id) {{
            const box = new THREE.Box3();
            let found = false;
            
            if (type === 'conductor') {{
                const extGroup = data.extensionGroups.find(g => g.includes(id)) || [id];
                conductorMeshes.forEach(m => {{
                    if (extGroup.includes(m.userData.id)) {{
                        if (!m.geometry.boundingBox) m.geometry.computeBoundingBox();
                        box.expandByObject(m);
                        found = true;
                    }}
                }});
            }} else if (type === 'pole') {{
                const m = poleMeshes.find(m => m.userData.id === id);
                if (m) {{
                    box.expandByObject(m);
                    found = true;
                }}
            }} else if (type === 'tree') {{
                const t = (data.trees || []).find(t => t.id === id);
                if (t) {{
                    const x = t.X ?? 0, y = t.Y ?? 0, z = t.Z ?? 0;
                    const h = t.height ?? 1, r = t.crown_radius ?? 1;
                    box.expandByPoint(new THREE.Vector3(x - r, y - r, z));
                    box.expandByPoint(new THREE.Vector3(x + r, y + r, z + h));
                    found = true;
                }}
            }} else if (type === 'building') {{
                const m = buildingMeshes.find(m => m.userData.id === id);
                if (m) {{
                    box.expandByObject(m);
                    found = true;
                }}
            }}
            
            if (found) {{
                const center = new THREE.Vector3();
                box.getCenter(center);
                const size = new THREE.Vector3();
                box.getSize(size);
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * (Math.PI / 180);
                let cameraDist = Math.abs(maxDim / 2 / Math.tan(fov / 2)) * 1.5;
                if (cameraDist < 10) cameraDist = 10;

                const direction = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
                if (direction.lengthSq() < 0.0001) direction.set(1, 1, 1).normalize();
                
                const newPos = center.clone().add(direction.multiplyScalar(cameraDist));
                controls.target.copy(center);
                camera.position.copy(newPos);
                controls.update();
            }}
        }}

        // Mouse Events
        renderer.domElement.addEventListener('click', (event) => {{
            // Shift + Click -> Lock/Unlock
            if (event.shiftKey && event.button === 0) {{
                mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
                mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
                raycaster.setFromCamera(mouse, camera);
                
                const objects = [...conductorMeshes, ...poleMeshes, ...buildingMeshes, ...vehicleMeshes, ...treeMeshes];
                const hits = raycaster.intersectObjects(objects, false);
                
                if (hits.length > 0) {{
                    const hit = hits[0];
                    const type = hit.object.userData.type;
                    const id = (type === 'tree' && hit.instanceId != null) ? hit.object.userData.ids[hit.instanceId] : hit.object.userData.id;
                    
                    if (lockedSelection && lockedSelection.type === type && lockedSelection.id === id) {{
                        // Unlock
                        lockedSelection = null;
                        previewSelection = null;
                        updateHighlights();
                        updateInfoPanel(null);
                        highlightItemInList(null, null);
                    }} else {{
                        // Lock
                        lockSelection(type, id);
                    }}
                }}
            }}
        }});

        renderer.domElement.addEventListener('pointermove', (event) => {{
            if (lockedSelection) return;
            
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            
            const objects = [...conductorMeshes, ...poleMeshes, ...buildingMeshes, ...vehicleMeshes, ...treeMeshes];
            const hits = raycaster.intersectObjects(objects, false);
            
            if (hits.length > 0) {{
                const hit = hits[0];
                const type = hit.object.userData.type;
                const id = (type === 'tree' && hit.instanceId != null) ? hit.object.userData.ids[hit.instanceId] : hit.object.userData.id;
                previewSelection = {{ type, id }};
                updateHighlights();
            }} else {{
                if (previewSelection) {{
                    previewSelection = null;
                    updateHighlights();
                }}
            }}
        }});

        renderer.domElement.addEventListener('dblclick', (event) => {{
            if (event.button !== 0) return;
            
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
            raycaster.setFromCamera(mouse, camera);
            const objects = [...conductorMeshes, ...poleMeshes, ...buildingMeshes, ...vehicleMeshes, ...treeMeshes];
            const hits = raycaster.intersectObjects(objects, false);

            if (hits.length > 0) {{
                controls.target.copy(hits[0].point);
            }} else {{
                // Reset view
                lockedSelection = null;
                previewSelection = null;
                updateHighlights();
                updateInfoPanel(null);
                
                if (data.poles.length > 0) {{
                    const first = data.poles[0];
                    const center = new THREE.Vector3(
                        (first.min[0] + first.max[0])/2,
                        (first.min[1] + first.max[1])/2,
                        (first.min[2] + first.max[2])/2
                    );
                    controls.target.copy(center);
                }}
            }}
        }});

        document.getElementById('openListBtn').onclick = () => {{
            const el = document.getElementById('sidebar');
            el.style.display = (el.style.display === 'none') ? 'flex' : 'none';
        }};

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Web interface saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interactive 3D web graph.")
    parser.add_argument("--json", help="Path to the JSON file or directory containing JSON files.")
    parser.add_argument("--output", help="Output HTML file (only used if --json is a single file).")
    parser.add_argument("--tol", type=float, default=1.0, help="Max distance at pole for extension.")
    parser.add_argument(
        "--pointcloud",
        help="Optional path to LAS/LAZ point cloud file with x,y,z and instance_id.",
        default=None,
    )
    
    args = parser.parse_args()
    
    if not args.json:
        parser.print_help()
        sys.exit(1)
        
    if os.path.isdir(args.json):
        # Process directory
        json_files = glob.glob(os.path.join(args.json, "*.json"))
        # Exclude those that are already output files
        json_files = [f for f in json_files if not (os.path.basename(f).startswith("web_") or os.path.basename(f).startswith("graph_"))]
        
        print(f"Processing {len(json_files)} JSON files in directory: {args.json}")
        for f in json_files:
            visualize_network_web(f, None, args.tol, args.pointcloud)
    else:
        # Process single file
        visualize_network_web(args.json, args.output, args.tol, args.pointcloud)
