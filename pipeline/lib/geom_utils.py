import numpy as np
import laspy
from typing import List, Dict, Tuple, Any, Optional
from shapely.geometry import MultiPoint, Polygon

from scipy.ndimage import binary_dilation
from skimage import measure

def _aabb_mesh(min_pt: np.ndarray, max_pt: np.ndarray) -> Dict[str, Any]:
    vertices = [
        [float(min_pt[0]), float(min_pt[1]), float(min_pt[2])],
        [float(max_pt[0]), float(min_pt[1]), float(min_pt[2])],
        [float(max_pt[0]), float(max_pt[1]), float(min_pt[2])],
        [float(min_pt[0]), float(max_pt[1]), float(min_pt[2])],
        [float(min_pt[0]), float(min_pt[1]), float(max_pt[2])],
        [float(max_pt[0]), float(min_pt[1]), float(max_pt[2])],
        [float(max_pt[0]), float(max_pt[1]), float(max_pt[2])],
        [float(min_pt[0]), float(max_pt[1]), float(max_pt[2])],
    ]
    faces = [
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
    ]
    return {"vertices": vertices, "faces": faces}

def compute_marching_cubes_mesh(
    points: np.ndarray,
    spacing: float = 0.5,
    padding: int = 2,
    dilate_iterations: int = 1,
) -> Dict[str, Any]:
    finite = np.all(np.isfinite(points), axis=1)
    points = np.asarray(points)[finite]
    if len(points) == 0: return {"vertices": [], "faces": []}
    min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)
    if len(points) == 1: return {"vertices": points.tolist(), "faces": []}
    if np.any(max_pt - min_pt < 1e-6): return _aabb_mesh(min_pt, max_pt)

    origin = min_pt - padding * spacing
    shape = np.ceil((max_pt - origin) / spacing).astype(int) + 1
    volume = np.zeros(np.maximum(shape, 3), dtype=np.uint8)
    idx = ((points - origin) / spacing).astype(int)
    volume[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    if dilate_iterations > 0:
        volume = binary_dilation(volume, iterations=dilate_iterations).astype(np.uint8)
    try:
        verts, faces, _, _ = measure.marching_cubes(volume, level=0.5, spacing=(spacing, spacing, spacing))
        world_verts = (verts + origin).tolist()
        return {"vertices": world_verts, "faces": faces.tolist()}
    except Exception:
        return _aabb_mesh(min_pt, max_pt)

def compute_obb(points: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Computes the 2D minimum rotated rectangle (OBB) of a set of 3D points.
    Returns:
        Dict with 'center' (3D), 'extent' (3D), 'angle' (rad), and 'footprint' (List of [x,y]).
    """
    if points.shape[0] < 3:
        # For very few points, default to AABB
        min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)
        center = (min_pt + max_pt) * 0.5
        extent = max_pt - min_pt
        return {
            "center": center.tolist(),
            "extent": extent.tolist(),
            "angle": 0.0,
            "footprint": [
                [float(min_pt[0]), float(min_pt[1])],
                [float(max_pt[0]), float(min_pt[1])],
                [float(max_pt[0]), float(max_pt[1])],
                [float(min_pt[0]), float(max_pt[1])],
                [float(min_pt[0]), float(min_pt[1])],
            ]
        }
    
    # 2D Projected points
    pts2d = points[:, :2]
    rect = MultiPoint(pts2d).minimum_rotated_rectangle
    
    if not isinstance(rect, Polygon):
        # Degenerate case (collinear points)
        min_pt, max_pt = np.min(points, axis=0), np.max(points, axis=0)
        center = (min_pt + max_pt) * 0.5
        extent = max_pt - min_pt
        return {
            "center": center.tolist(),
            "extent": extent.tolist(),
            "angle": 0.0,
            "footprint": [
                [float(min_pt[0]), float(min_pt[1])],
                [float(max_pt[0]), float(min_pt[1])],
                [float(max_pt[0]), float(max_pt[1])],
                [float(min_pt[0]), float(max_pt[1])],
                [float(min_pt[0]), float(min_pt[1])],
            ]
        }

    # Extract polygon coordinates
    coords = np.array(rect.exterior.coords)
    # The minimum_rotated_rectangle for a Polygon has 5 coords (closing point)
    # Let's derive the principal axes from the longest side
    edges = coords[1:] - coords[:-1]
    edge_lengths = np.linalg.norm(edges, axis=1)
    
    # The longest side defines the local X axis
    long_idx = np.argmax(edge_lengths)
    v_long = edges[long_idx]
    angle = float(np.arctan2(v_long[1], v_long[0]))
    
    # Dimensions along local axes
    lx = float(edge_lengths[long_idx])
    # The perpendicular side's length (the next or previous edge)
    ly = float(edge_lengths[(long_idx + 1) % 4])
    
    z_min, z_max = float(np.min(points[:, 2])), float(np.max(points[:, 2]))
    lz = z_max - z_min
    
    # Center calculation
    cx, cy = rect.centroid.x, rect.centroid.y
    cz = (z_min + z_max) * 0.5
    
    return {
        "center": [float(cx), float(cy), float(cz)],
        "extent": [lx, ly, lz],
        "angle": angle,
        "footprint": coords.tolist()
    }

def dbscan_obb(
    obbs: List[Optional[Dict[str, Any]]],
    mins: List[np.ndarray],
    maxs: List[np.ndarray],
    ids: List[int],
    eps: float,
    min_samples: int = 1,
) -> List[List[int]]:
    """DBSCAN clustering using OBB (Shapely polygon) distances with AABB fallback."""
    n = len(ids)
    if n == 0:
        return []

    polys: List[Optional[Polygon]] = []
    for obb in obbs:
        if obb and "footprint" in obb:
            try:
                polys.append(Polygon(obb["footprint"]))
            except Exception:
                polys.append(None)
        else:
            polys.append(None)

    mins_arr = np.array(mins)
    maxs_arr = np.array(maxs)

    neighbors: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = polys[i], polys[j]
            if p1 is not None and p2 is not None:
                d = p1.distance(p2)
            else:
                d = aabb_min_distance(mins_arr[i], maxs_arr[i], mins_arr[j], maxs_arr[j])
            if d <= eps:
                neighbors[i].append(j)
                neighbors[j].append(i)

    visited = [False] * n
    labels = [-1] * n
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) + 1 < min_samples:
            continue
        labels[i] = cluster_id
        seeds = list(neighbors[i])
        seed_set = set(seeds)
        idx = 0
        while idx < len(seeds):
            q = seeds[idx]
            idx += 1
            if not visited[q]:
                visited[q] = True
                if len(neighbors[q]) + 1 >= min_samples:
                    for ni in neighbors[q]:
                        if ni not in seed_set:
                            seed_set.add(ni)
                            seeds.append(ni)
            if labels[q] == -1:
                labels[q] = cluster_id
        cluster_id += 1

    clusters_by_id: Dict[int, List[int]] = {}
    for idx_val, lbl in enumerate(labels):
        if lbl != -1:
            clusters_by_id.setdefault(lbl, []).append(ids[idx_val])
    return list(clusters_by_id.values())


def load_laz_points(laz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load points from a LAZ file.
    Returns:
        points: (N, 3) float array of XYZ coordinates
        sem_classes: (N,) int array of semantic classes
        ins_classes: (N,) int array of instance IDs (dataset convention: unique across all classes in the tile)

    Pipeline code uses ins_classes directly as graph object ids for buildings/vehicles/trees.
    """
    las = laspy.read(laz_path)
    if 'classification' not in las.point_format.dimension_names:
        return np.array([]), np.array([]), np.array([])
    
    xyz = np.vstack((las.x, las.y, las.z)).transpose()
    sem = np.array(las.classification)
    # Use instance dimension if it exists, otherwise return all -1
    if 'instance' in las.point_format.dimension_names:
        ins = np.array(las.instance)
    else:
        ins = np.full(len(sem), -1, dtype=int)

    # Drop points with NaN/Inf
    finite = np.all(np.isfinite(xyz), axis=1)
    if not np.all(finite):
        xyz, sem, ins = xyz[finite], sem[finite], ins[finite]
    return xyz, sem, ins

def aabb_min_distance(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray) -> float:
    """
    Compute the minimum distance between two axis-aligned bounding boxes (AABBs).
    If they intersect, the distance is 0.0.
    """
    dx = max(0.0, min2[0] - max1[0], min1[0] - max2[0])
    dy = max(0.0, min2[1] - max1[1], min1[1] - max2[1])
    dz = max(0.0, min2[2] - max1[2], min1[2] - max2[2])
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))

def dbscan_aabb(
    mins: List[np.ndarray], 
    maxs: List[np.ndarray], 
    ids: List[int], 
    eps: float, 
    min_samples: int = 1
) -> List[List[int]]:
    """
    Cluster AABBs using DBSCAN logic based on minimum AABB distance.
    
    Args:
        mins: List of min coordinates (each np.ndarray or list of floats)
        maxs: List of max coordinates (each np.ndarray or list of floats)
        ids: List of object IDs corresponding to the AABBs
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
                     Defaults to 1, effectively treating connected components as clusters (single-linkage) if min_samples=1.
                     Note: The point itself counts towards min_samples.
    
    Returns:
        A list of clusters, where each cluster is a list of object IDs.
        Noise points (if any, when min_samples > 1) are not included in the returned clusters.
    """
    n = len(ids)
    if n == 0:
        return []

    # 1. Compute neighbors (vectorized in chunks to avoid large memory allocations)
    neighbors: List[List[int]] = [[] for _ in range(n)]
    mins_arr = np.array(mins)
    maxs_arr = np.array(maxs)
    
    chunk_size = 2000
    eps_sq = eps * eps
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_m1 = mins_arr[i:end_i]
        chunk_M1 = maxs_arr[i:end_i]
        
        dx = np.maximum(0.0, np.maximum(mins_arr[None, :, 0] - chunk_M1[:, None, 0], chunk_m1[:, None, 0] - maxs_arr[None, :, 0]))
        dy = np.maximum(0.0, np.maximum(mins_arr[None, :, 1] - chunk_M1[:, None, 1], chunk_m1[:, None, 1] - maxs_arr[None, :, 1]))
        dz = np.maximum(0.0, np.maximum(mins_arr[None, :, 2] - chunk_M1[:, None, 2], chunk_m1[:, None, 2] - maxs_arr[None, :, 2]))
        
        dists_sq = dx*dx + dy*dy + dz*dz
        mask = dists_sq <= eps_sq
        
        row_idx, col_idx = np.nonzero(mask)
        abs_row_idx = row_idx + i
        
        # Keep only upper triangle (j > i) to avoid duplicate edges and self-loops
        valid = col_idx > abs_row_idx
        abs_row_idx = abs_row_idx[valid]
        col_idx = col_idx[valid]
        
        for r, c in zip(abs_row_idx, col_idx):
            neighbors[r].append(c)
            neighbors[c].append(r)
                
    # 2. DBSCAN Clustering
    visited = [False] * n
    labels = [-1] * n
    cluster_id = 0
    
    for i in range(n):
        if visited[i]:
            continue
            
        visited[i] = True
        neighbor_indices = neighbors[i]
        
        # Determine if core point
        if len(neighbor_indices) + 1 < min_samples:
            # Mark as Noise
            labels[i] = -1
        else:
            # Start a new cluster
            labels[i] = cluster_id
            
            # Expand cluster
            seeds = list(neighbor_indices)
            seed_set_lookup = set(seeds) # For O(1) lookup to avoid duplicates in seeds list
            
            idx = 0
            while idx < len(seeds):
                q = seeds[idx]
                idx += 1
                
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = neighbors[q]
                    
                    if len(q_neighbors) + 1 >= min_samples:
                        for n_idx in q_neighbors:
                            if n_idx not in seed_set_lookup:
                                seed_set_lookup.add(n_idx)
                                seeds.append(n_idx)
                
                # If point was noise or undefined, it joins the cluster
                if labels[q] == -1:
                    labels[q] = cluster_id
            
            cluster_id += 1
            
    # Collect clusters by ID
    clusters_by_id: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        if lbl != -1:
            if lbl not in clusters_by_id:
                clusters_by_id[lbl] = []
            clusters_by_id[lbl].append(ids[idx])
            
    return list(clusters_by_id.values())
