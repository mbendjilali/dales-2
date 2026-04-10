"""
Unified graph edges: every relation is {id, a_type, b_type, a_id, b_id, class}.
Types: building, conductor, pole, tree, vehicle.
Classes: extension, bifurcation, cross, support, support_building, adjacent, near.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

# Lexicographic order for canonical undirected endpoints
_TYPE_ORDER = ("building", "conductor", "pole", "tree", "vehicle")

PROXIMITY_EDGE_CLASSES = frozenset({"adjacent", "near"})


def _type_rank(t: str) -> int:
    t = str(t).lower()
    try:
        return _TYPE_ORDER.index(t)
    except ValueError:
        return len(_TYPE_ORDER)


def canonical_edge(
    a_type: str, a_id: int, b_type: str, b_id: int
) -> Tuple[str, int, str, int]:
    """Undirected canonical order: lower (type, id) first."""
    ta, tb = str(a_type).lower(), str(b_type).lower()
    ia, ib = int(a_id), int(b_id)
    ra, rb = _type_rank(ta), _type_rank(tb)
    if ra < rb or (ra == rb and ia <= ib):
        return ta, ia, tb, ib
    return tb, ib, ta, ia


def ensure_edges_list(graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    edges = graph_data.get("edges")
    if edges is None:
        edges = []
        graph_data["edges"] = edges
    return edges


def next_edge_id(edges: List[Dict[str, Any]]) -> int:
    if not edges:
        return 0
    return max(int(e.get("id", 0)) for e in edges) + 1


def strip_proximity_edges(graph_data: Dict[str, Any]) -> None:
    """Remove adjacent/near edges (for recomputation)."""
    edges = graph_data.get("edges") or []
    graph_data["edges"] = [
        e
        for e in edges
        if str(e.get("class", "")).lower() not in PROXIMITY_EDGE_CLASSES
    ]


def delete_proximity_edges_for_canonical_pair(
    graph_data: Dict[str, Any],
    ca: str,
    cia: int,
    cb: str,
    cib: int,
) -> int:
    """
    Remove all adjacent/near edges matching this undirected canonical pair.
    Used so at most one proximity relation exists per pair (see dedupe_proximity_edges).
    """
    ca, cia, cb, cib = canonical_edge(ca, cia, cb, cib)
    ca, cb = str(ca).lower(), str(cb).lower()
    edges = graph_data.get("edges") or []
    kept: List[Dict[str, Any]] = []
    removed = 0
    for e in edges:
        cls_e = str(e.get("class", "")).lower()
        if cls_e not in PROXIMITY_EDGE_CLASSES:
            kept.append(e)
            continue
        if (
            str(e.get("a_type", "")).lower() == ca
            and int(e.get("a_id", -1)) == cia
            and str(e.get("b_type", "")).lower() == cb
            and int(e.get("b_id", -1)) == cib
        ):
            removed += 1
            continue
        kept.append(e)
    graph_data["edges"] = kept
    return removed


def dedupe_proximity_edges(graph_data: Dict[str, Any]) -> None:
    """
    Collapse duplicate proximity edges for the same undirected pair.
    If both adjacent and near exist (e.g. after threshold changes without a full strip),
    keep adjacent. Idempotent.
    """
    edges = graph_data.get("edges") or []
    nonprox: List[Dict[str, Any]] = []
    prox: List[Dict[str, Any]] = []
    for e in edges:
        if str(e.get("class", "")).lower() in PROXIMITY_EDGE_CLASSES:
            prox.append(e)
        else:
            nonprox.append(e)
    by_key: Dict[Tuple[str, int, str, int], List[Dict[str, Any]]] = {}
    for e in prox:
        try:
            ca, cia, cb, cib = canonical_edge(
                str(e.get("a_type", "")),
                int(e.get("a_id", -1)),
                str(e.get("b_type", "")),
                int(e.get("b_id", -1)),
            )
        except (TypeError, ValueError):
            nonprox.append(e)
            continue
        by_key.setdefault((ca, cia, cb, cib), []).append(e)
    kept_prox: List[Dict[str, Any]] = []
    for group in by_key.values():
        if len(group) == 1:
            kept_prox.append(group[0])
            continue
        adjacent = [x for x in group if str(x.get("class", "")).lower() == "adjacent"]
        kept_prox.append(adjacent[0] if adjacent else group[0])
    graph_data["edges"] = nonprox + kept_prox


def upsert_unified_edge(
    graph_data: Dict[str, Any],
    a_type: str,
    a_id: int,
    b_type: str,
    b_id: int,
    cls: str,
) -> Dict[str, Any]:
    """Create or update edge (match on canonical endpoints + class)."""
    edges = ensure_edges_list(graph_data)
    cls_l = str(cls).lower()
    ca, cia, cb, cib = canonical_edge(a_type, a_id, b_type, b_id)
    for e in edges:
        if (
            str(e.get("a_type", "")).lower() == ca
            and int(e.get("a_id", -1)) == cia
            and str(e.get("b_type", "")).lower() == cb
            and int(e.get("b_id", -1)) == cib
            and str(e.get("class", "")).lower() == cls_l
        ):
            return e
    e = {
        "id": next_edge_id(edges),
        "a_type": ca,
        "a_id": cia,
        "b_type": cb,
        "b_id": cib,
        "class": cls_l,
    }
    edges.append(e)
    return e


def delete_edges_for_endpoints(
    graph_data: Dict[str, Any],
    obj_type: str,
    obj_ids: Set[int],
) -> None:
    """Remove edges where either endpoint matches type+id."""
    ot = str(obj_type).lower()
    edges = graph_data.get("edges") or []
    graph_data["edges"] = [
        e
        for e in edges
        if not (
            (
                str(e.get("a_type", "")).lower() == ot
                and int(e.get("a_id", -1)) in obj_ids
            )
            or (
                str(e.get("b_type", "")).lower() == ot
                and int(e.get("b_id", -1)) in obj_ids
            )
        )
    ]


def conductor_pole_ids_from_edges(
    graph_data: Dict[str, Any], conductor_endpoint_id: int
) -> List[int]:
    """Match conductor side of support edges by conductor LAZ instance id (see rewire_graph_to_laz_instance_ids)."""
    out: List[int] = []
    cid = int(conductor_endpoint_id)
    for e in graph_data.get("edges") or []:
        if str(e.get("class", "")).lower() != "support":
            continue
        at, bt = str(e.get("a_type", "")).lower(), str(e.get("b_type", "")).lower()
        if at == "conductor" and int(e.get("a_id", -1)) == cid and bt == "pole":
            out.append(int(e["b_id"]))
        elif bt == "conductor" and int(e.get("b_id", -1)) == cid and at == "pole":
            out.append(int(e["a_id"]))
    return sorted(set(out))


def conductor_support_building_ids_from_edges(
    graph_data: Dict[str, Any], conductor_endpoint_id: int
) -> List[int]:
    out: List[int] = []
    cid = int(conductor_endpoint_id)
    for e in graph_data.get("edges") or []:
        if str(e.get("class", "")).lower() != "support_building":
            continue
        at, bt = str(e.get("a_type", "")).lower(), str(e.get("b_type", "")).lower()
        if at == "conductor" and int(e.get("a_id", -1)) == cid and bt == "building":
            out.append(int(e["b_id"]))
        elif bt == "conductor" and int(e.get("b_id", -1)) == cid and at == "building":
            out.append(int(e["a_id"]))
    return sorted(set(out))


def rewire_graph_to_laz_instance_ids(
    graph_data: Dict[str, Any],
    geom_data: Dict[str, Any],
    laz_data: Optional[Tuple[Any, Any, Any]] = None,
) -> None:
    """
    After add_conductor_instances / add_pole_instances (optional):
    - Mark virtual poles (no PCL pole geometry); they never take LAZ pole votes.
    - Assign conductor / virtual-pole / colliding-pole ids using next free integers outside the
      PCL instance universe and not conflicting buildings / vehicles / trees / other graph ids.
    - Remap poles[].id and geom pole keys; rewrite edges; tag pole-involved edges with virtual_pole.
    - Remove conductor uid; dedupe edges and reassign sequential edge ids.
    """
    from pipeline.lib.instance_ids import (
        laz_used_instance_ids,
        mark_virtual_poles_from_geom,
        next_free_tile_id,
    )

    mark_virtual_poles_from_geom(graph_data, geom_data)

    poles = graph_data.get("poles") or []
    conductors = graph_data.get("conductors") or []

    pcl_used = laz_used_instance_ids(laz_data) if laz_data is not None else set()

    building_ids = {
        int(b["id"]) for b in graph_data.get("buildings") or [] if b.get("id") is not None
    }
    vehicle_ids = {
        int(v["id"]) for v in graph_data.get("vehicles") or [] if v.get("id") is not None
    }
    tree_ids = {int(t["id"]) for t in graph_data.get("trees") or [] if t.get("id") is not None}
    bvgt = building_ids | vehicle_ids | tree_ids

    occupied_graph: Set[int] = set(building_ids) | set(vehicle_ids) | set(tree_ids)

    def cond_sort_key(c: Dict[str, Any]) -> Tuple[int, int, int]:
        u = c.get("uid")
        if u is not None:
            return (0, int(u), 0)
        return (1, int(c.get("link_idx") or 0), int(c.get("conductor_id") or 0))

    for c in sorted(conductors, key=cond_sort_key):
        inst = c.get("instance_id")
        need_new = inst is None
        if inst is not None:
            ii = int(inst)
            if ii in bvgt or ii in occupied_graph:
                need_new = True
        if need_new:
            nid = next_free_tile_id(pcl_used, occupied_graph)
            c["instance_id"] = nid
            occupied_graph.add(nid)
        else:
            occupied_graph.add(int(inst))

    uid_to_inst: Dict[int, int] = {}
    for c in conductors:
        u = c.get("uid")
        inst = c.get("instance_id")
        if u is not None and inst is not None:
            uid_to_inst[int(u)] = int(inst)

    conductor_inst = {
        int(c["instance_id"]) for c in conductors if c.get("instance_id") is not None
    }
    conflict_for_pole = bvgt | conductor_inst

    pole_old_to_new: Dict[int, int] = {}
    pole_claimed: Set[int] = set()
    virtual_pole_ids: Set[int] = set()

    poles_sorted = sorted(
        (p for p in poles if p.get("id") is not None),
        key=lambda p: int(p["id"]),
    )
    for pole in poles_sorted:
        old_id = int(pole["id"])
        is_virt = bool(pole.get("is_virtual_pole"))

        if is_virt:
            pole.pop("instance_id", None)
            nid = next_free_tile_id(pcl_used, occupied_graph)
            occupied_graph.add(nid)
            pole["id"] = nid
            pole["instance_id"] = nid
            pole_old_to_new[old_id] = nid
            pole_claimed.add(nid)
            virtual_pole_ids.add(nid)
            continue

        prefer = pole.get("instance_id")
        chosen: Optional[int] = None
        if prefer is not None:
            pi = int(prefer)
            if pi not in conflict_for_pole and pi not in pole_claimed:
                chosen = pi

        if chosen is None:
            nid = next_free_tile_id(pcl_used, occupied_graph)
            occupied_graph.add(nid)
            chosen = nid

        pole["id"] = chosen
        pole["instance_id"] = chosen
        pole_old_to_new[old_id] = chosen
        pole_claimed.add(chosen)

    gp = geom_data.get("poles") or {}
    new_gp: Dict[str, Any] = {}
    for k, v in gp.items():
        try:
            oid = int(str(k).strip())
        except (TypeError, ValueError):
            new_gp[str(k)] = v
            continue
        nid = pole_old_to_new.get(oid, oid)
        new_gp[str(nid)] = v
    geom_data["poles"] = new_gp

    for c in conductors:
        plist = c.get("poles")
        if not plist:
            continue
        c["poles"] = [pole_old_to_new.get(int(p), int(p)) for p in plist]

    def map_endpoint(t: str, idv: int) -> Tuple[str, int]:
        tl = str(t).lower()
        i = int(idv)
        if tl == "conductor":
            return tl, uid_to_inst.get(i, i)
        if tl == "pole":
            return tl, pole_old_to_new.get(i, i)
        return tl, i

    def edge_has_virtual_pole(
        ca: str, cia: int, cb: str, cib: int
    ) -> bool:
        return (ca == "pole" and cia in virtual_pole_ids) or (
            cb == "pole" and cib in virtual_pole_ids
        )

    edges = graph_data.get("edges") or []
    edge_by_key: Dict[Tuple[str, int, str, int, str], Dict[str, Any]] = {}
    for e in edges:
        cls_l = str(e.get("class", "")).lower()
        at = str(e.get("a_type", ""))
        bt = str(e.get("b_type", ""))
        try:
            aia = int(e.get("a_id"))
            bib = int(e.get("b_id"))
        except (TypeError, ValueError):
            continue
        na_t, na_i = map_endpoint(at, aia)
        nb_t, nb_i = map_endpoint(bt, bib)
        ca, cia, cb, cib = canonical_edge(na_t, na_i, nb_t, nb_i)
        key = (ca, cia, cb, cib, cls_l)
        virt = edge_has_virtual_pole(ca, cia, cb, cib)
        if key not in edge_by_key:
            edge_by_key[key] = {
                "id": 0,
                "a_type": ca,
                "a_id": cia,
                "b_type": cb,
                "b_id": cib,
                "class": cls_l,
            }
            if virt:
                edge_by_key[key]["virtual_pole"] = True
        elif virt:
            edge_by_key[key]["virtual_pole"] = True

    new_edges = list(edge_by_key.values())
    for i, ed in enumerate(new_edges):
        ed["id"] = i
    graph_data["edges"] = new_edges

    for c in conductors:
        c.pop("uid", None)


def to_frontend_group_relations(graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Viewer/API shape: member_type, group_id, a_id, b_id, class, optional peer_type.
    """
    out: List[Dict[str, Any]] = []
    buildings = {b["id"]: b for b in graph_data.get("buildings") or []}
    vehicles = {v["id"]: v for v in graph_data.get("vehicles") or []}

    for e in graph_data.get("edges") or []:
        cls = str(e.get("class", "")).lower()
        if cls not in PROXIMITY_EDGE_CLASSES:
            continue
        at = str(e.get("a_type", "")).lower()
        bt = str(e.get("b_type", "")).lower()
        aid = int(e.get("a_id", -1))
        bid = int(e.get("b_id", -1))
        rid = int(e.get("id", 0))

        if at == bt == "building":
            ga = buildings.get(aid, {}).get("group_id")
            gb = buildings.get(bid, {}).get("group_id")
            gid = ga if ga is not None else (gb if gb is not None else 0)
            ao, bo = sorted((aid, bid))
            out.append(
                {
                    "id": rid,
                    "member_type": "building",
                    "group_id": int(gid) if gid is not None else 0,
                    "a_id": ao,
                    "b_id": bo,
                    "class": cls,
                }
            )
        elif at == bt == "vehicle":
            ga = vehicles.get(aid, {}).get("group_id")
            gb = vehicles.get(bid, {}).get("group_id")
            gid = ga if ga is not None else (gb if gb is not None else 0)
            ao, bo = sorted((aid, bid))
            out.append(
                {
                    "id": rid,
                    "member_type": "vehicle",
                    "group_id": int(gid) if gid is not None else 0,
                    "a_id": ao,
                    "b_id": bo,
                    "class": cls,
                }
            )
        elif at == "tree" and bt in ("building", "vehicle", "pole", "conductor"):
            out.append(
                {
                    "id": rid,
                    "member_type": "tree",
                    "group_id": 0,
                    "peer_type": bt,
                    "a_id": aid,
                    "b_id": bid,
                    "class": cls,
                }
            )
        elif bt == "tree" and at in ("building", "vehicle", "pole", "conductor"):
            out.append(
                {
                    "id": rid,
                    "member_type": "tree",
                    "group_id": 0,
                    "peer_type": at,
                    "a_id": bid,
                    "b_id": aid,
                    "class": cls,
                }
            )
    return out
