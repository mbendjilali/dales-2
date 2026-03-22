"""
Unified graph edges: every relation is {id, a_type, b_type, a_id, b_id, class}.
Types: building, conductor, pole, tree, vehicle.
Classes: extension, bifurcation, cross, support, support_building, adjacent, near.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

# Lexicographic order for canonical undirected endpoints
_TYPE_ORDER = ("building", "conductor", "pole", "tree", "vehicle")

PROXIMITY_EDGE_CLASSES = frozenset({"adjacent", "near"})

# Used when no LAZ powerline match; kept below typical LAZ instance ranges.
SYNTHETIC_CONDUCTOR_INSTANCE_ID_BASE = 2_000_000_000


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
    graph_data: Dict[str, Any], geom_data: Dict[str, Any]
) -> None:
    """
    After add_conductor_instances / add_pole_instances (optional):
    - Assign synthetic conductor instance_id when missing (from numeric uid).
    - Set poles[].id to instance_id when uniquely mappable.
    - Remap geom_data[\"poles\"] keys to new pole ids.
    - Remap conductor support pole lists.
    - Rewrite edges: conductor endpoints use instance_id; pole endpoints use new pole id.
    - Remove conductor \"uid\"; deduplicate edges and reassign sequential ids.
    Idempotent if edges already use instance ids (no uid left on conductors).
    """
    poles = graph_data.get("poles") or []
    conductors = graph_data.get("conductors") or []

    base = SYNTHETIC_CONDUCTOR_INSTANCE_ID_BASE
    for c in conductors:
        if c.get("instance_id") is not None:
            continue
        u = c.get("uid")
        if u is None:
            continue
        c["instance_id"] = base + int(u)

    uid_to_inst: Dict[int, int] = {}
    for c in conductors:
        u = c.get("uid")
        inst = c.get("instance_id")
        if u is None or inst is None:
            continue
        uid_to_inst[int(u)] = int(inst)

    new_id_to_old: Dict[int, int] = {}
    pole_old_to_new: Dict[int, int] = {}
    for pole in poles:
        if pole.get("id") is None:
            continue
        old_id = int(pole["id"])
        inst = pole.get("instance_id")
        if inst is None:
            continue
        new_id = int(inst)
        prev_old = new_id_to_old.get(new_id)
        if prev_old is not None and prev_old != old_id:
            pole.pop("instance_id", None)
            continue
        new_id_to_old[new_id] = old_id
        pole_old_to_new[old_id] = new_id
        pole["id"] = new_id

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

    edges = graph_data.get("edges") or []
    new_edges: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int, str, int, str]] = set()
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
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(
            {
                "id": 0,
                "a_type": ca,
                "a_id": cia,
                "b_type": cb,
                "b_id": cib,
                "class": cls_l,
            }
        )
    for i, e in enumerate(new_edges):
        e["id"] = i
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
