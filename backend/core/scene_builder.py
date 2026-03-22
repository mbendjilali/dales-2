import numpy as np
import colorsys
from typing import List, Dict, Any, Set

from backend.core.graph_edges import (
    to_frontend_group_relations,
    conductor_pole_ids_from_edges,
    conductor_support_building_ids_from_edges,
)


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
    sat = 0.8
    val = 0.9
    rgb = colorsys.hsv_to_rgb(hue, sat, val)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


GROUP_TYPES = ("trees", "buildings", "vehicles", "poles", "conductors")


def _groups_key(object_type: str) -> str:
    return object_type.rstrip("s") + "_groups" if object_type.endswith("s") else object_type + "_groups"


def build_scene_data(graph_data: Dict[str, Any], geom_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transforms raw graph and geometry data into a format suitable for the frontend viewer."""

    # 1. Merge geometry into graph objects
    b_hulls = geom_data.get("buildingHulls", {})
    b_geoms = geom_data.get("buildings", {})
    buildings = graph_data.get("buildings", [])
    for b in buildings:
        bid_str = str(b["id"])
        if bid_str in b_geoms:
            geom = b_geoms[bid_str]
            b["min"] = geom.get("min")
            b["max"] = geom.get("max")
        if bid_str in b_hulls:
            hull_data = b_hulls[bid_str]
            q_verts = hull_data.get("vertices", [])
            verts = [[float(v[0]), float(v[1]), float(v[2])] for v in q_verts]
            b["hull"] = {"vertices": verts, "faces": hull_data.get("faces", [])}

    v_hulls = geom_data.get("vehicleHulls", {})
    v_geoms = geom_data.get("vehicles", {})
    vehicles = graph_data.get("vehicles", [])
    for v in vehicles:
        vid_str = str(v["id"])
        if vid_str in v_geoms:
            geom = v_geoms[vid_str]
            v["min"] = geom.get("min")
            v["max"] = geom.get("max")
        if vid_str in v_hulls:
            hull_data = v_hulls[vid_str]
            q_verts = hull_data.get("vertices", [])
            verts = [[float(v[0]), float(v[1]), float(v[2])] for v in q_verts]
            v["hull"] = {"vertices": verts, "faces": hull_data.get("faces", [])}

    t_geoms = geom_data.get("trees", {})
    trees = graph_data.get("trees", [])
    for t in trees:
        tid_str = str(t["id"])
        if tid_str in t_geoms:
            geom = t_geoms[tid_str]
            for key in ["X", "Y", "Z", "height", "crown_radius", "min", "max"]:
                t[key] = geom.get(key)

    # 2. Process poles
    js_poles = []
    min_z_candidates = []
    p_geoms = geom_data.get("poles", {})
    for pole in graph_data.get('poles', []):
        pid_str = str(pole.get("id"))
        geom = p_geoms.get(pid_str, {})
        for key in ["X", "Y", "Z", "footprint"]:
            pole[key] = geom.get(key)
        fp = pole.get('footprint')
        if fp and isinstance(fp.get('min'), list) and len(fp['min']) >= 3:
            min_z_candidates.append(float(fp['min'][2]))
        elif 'Z' in pole and pole['Z'] is not None:
            min_z_candidates.append(float(pole['Z']))
    global_min_z = min(min_z_candidates) if min_z_candidates else 0.0

    GROUND_POLE_MAX_HEIGHT = 1.0

    for pole in graph_data.get('poles', []):
        fp = pole.get('footprint')
        if not fp:
            x, y, z = pole.get('X'), pole.get('Y'), pole.get('Z')
            if x is None or y is None or z is None:
                continue
            half = 0.3
            fp = {
                'min': [x - half, y - half, global_min_z],
                'max': [x + half, y + half, z],
                'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                'is_virtual': True,
            }

        is_virtual = bool(fp.get('is_virtual', False))
        height = None
        if isinstance(fp.get('min'), list) and isinstance(fp.get('max'), list) and len(fp['min']) >= 3 and len(fp['max']) >= 3:
            try:
                height = float(fp['max'][2]) - float(fp['min'][2])
            except Exception:
                height = None
        is_ground = bool(is_virtual and height is not None and height <= GROUND_POLE_MAX_HEIGHT)
        pole['is_ground'] = is_ground
        is_building_support = pole.get('is_building_support', False)

        pole_data = {
            'min': fp['min'], 'max': fp['max'],
            'is_virtual': is_virtual, 'is_building_support': is_building_support,
            'is_ground': is_ground, 'id': pole.get('id'),
        }
        if not is_virtual:
            pole_data['position'] = [pole.get('X'), pole.get('Y'), pole.get('Z')]
            pole_data['rotation'] = fp.get('rotation')
        js_poles.append(pole_data)

    # 3. Process conductors
    js_conductors: List[Dict[str, Any]] = []
    comp_id_to_scene_ids: Dict[int, List[str]] = {}
    conductor_edge_id_to_scene_id: Dict[int, str] = {}

    conductors = graph_data.get('conductors', [])
    c_geoms = geom_data.get('conductors', {})
    max_comp = max((x.get('component', 0) for x in conductors), default=1)
    if max_comp < 1:
        max_comp = 1

    for c in conductors:
        geom_key = f"{c['link_idx']}_{c['conductor_id']}"
        iid = c["instance_id"]
        scene_id = str(int(iid))
        geom = c_geoms.get(geom_key, {})
        for key in ["model", "startpoint", "endpoint"]:
            c[key] = geom.get(key)
        points = generate_conductor_curve(c, steps=20)
        comp_idx = c.get('component', 0)
        color = get_distinct_color(index=comp_idx, total=max_comp)
        js_conductors.append({
            'points': points,
            'color': color,
            'id': scene_id,
            'component': comp_idx,
        })
        comp_id_to_scene_ids.setdefault(comp_idx, []).append(scene_id)
        conductor_edge_id_to_scene_id[int(iid)] = scene_id

    extension_groups = list(comp_id_to_scene_ids.values())

    conductor_extensions_js: Dict[str, List[str]] = {}
    bifurcation_map_js: Dict[str, List[str]] = {}
    crosses_map_js: Dict[str, List[str]] = {}

    def _add_cond_pair(m: Dict[str, List[str]], sa: str, sb: str) -> None:
        if sa == sb:
            return
        m.setdefault(sa, []).append(sb)
        m.setdefault(sb, []).append(sa)

    for e in graph_data.get("edges") or []:
        cls = str(e.get("class", "")).lower()
        if cls not in ("extension", "bifurcation", "cross"):
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
        if cls == "extension":
            _add_cond_pair(conductor_extensions_js, sa, sb)
        elif cls == "bifurcation":
            _add_cond_pair(bifurcation_map_js, sa, sb)
        else:
            _add_cond_pair(crosses_map_js, sa, sb)

    support_pole_map: Dict[str, List[int]] = {}
    support_building_map: Dict[str, List[int]] = {}
    for c in conductors:
        iid = int(c["instance_id"])
        scene_id = str(iid)
        support_pole_map[scene_id] = conductor_pole_ids_from_edges(graph_data, iid)
        support_building_map[scene_id] = conductor_support_building_ids_from_edges(
            graph_data, iid
        )

    ground_pole_ids: Set[int] = set()
    for pole in graph_data.get('poles', []):
        pid = pole.get('id')
        if pid is not None and pole.get('is_ground'):
            ground_pole_ids.add(int(pid))

    support_ground_map: Dict[str, List[int]] = {}
    for cond_scene_id, pole_ids in support_pole_map.items():
        for pid in pole_ids:
            if int(pid) in ground_pole_ids:
                support_ground_map.setdefault(cond_scene_id, []).append(int(pid))

    pole_to_conductors: Dict[int, List[str]] = {}
    for cond_scene_id, pole_ids in support_pole_map.items():
        for pid in pole_ids:
            pole_to_conductors.setdefault(pid, []).append(cond_scene_id)

    building_to_support_conductors: Dict[int, List[str]] = {}
    for cond_scene_id, building_ids in support_building_map.items():
        for bid in building_ids:
            building_to_support_conductors.setdefault(bid, []).append(cond_scene_id)

    BUILDING_COLOR_NONE = 0x888888
    BUILDING_COLOR_SUPPORT = 0x1565c0
    building_default_colors: Dict[int, int] = {}
    building_default_opacity: Dict[int, float] = {}
    for b in buildings:
        bid = b['id']
        is_support = bid in building_to_support_conductors
        building_default_colors[bid] = BUILDING_COLOR_SUPPORT if is_support else BUILDING_COLOR_NONE
        building_default_opacity[bid] = 0.6 if is_support else 0.3

    # 5. Construct scene
    return {
        'poles': js_poles,
        'conductors': js_conductors,
        'conductorRadius': 0.1,
        'extensionGroups': extension_groups,
        'conductorExtensions': conductor_extensions_js,
        'bifurcations': bifurcation_map_js,
        'crosses': crosses_map_js,
        'supportPoles': support_pole_map,
        'supportBuildings': support_building_map,
        'supportGrounds': support_ground_map,
        'poleToConductors': pole_to_conductors,
        'buildingToSupportConductors': building_to_support_conductors,
        'buildingDefaultColors': building_default_colors,
        'buildingDefaultOpacity': building_default_opacity,
        'buildings': buildings,
        'vehicles': vehicles,
        'trees': graph_data.get('trees', []),
        'buildingGroups': graph_data.get('building_groups', []),
        'vehicleGroups': graph_data.get('vehicle_groups', []),
        'treeGroups': graph_data.get('tree_groups', []),
        'groupRelations': to_frontend_group_relations(graph_data),
    }
