# Network Reasoning

A scene graph construction and editing tool for classified 3D point clouds. It processes LiDAR data (LAZ/LAS) with semantic and instance labels to produce structured JSON graphs capturing spatial topology and inter-object relations. An integrated web-based 3D viewer enables interactive inspection, grouping, and relation editing.

## Project Structure

```
backend/                  FastAPI server and core domain logic
  main.py                 API entry point (endpoints, LAZ upload)
  core/
    graph_manager.py      Tile loading, grouping, relation computation
    scene_builder.py      Transforms graph + geometry into frontend-ready data

pipeline/                 Batch processing pipeline
  build_full_graph.py     Orchestrator: network → graph + geometry for all tiles
  lib/
    generate_json_graph.py  Pole/conductor/extension graph from network JSON
    add_buildings.py        Building extraction and OBB-based clustering
    add_vehicules.py        Vehicle extraction and OBB-based clustering
    add_trees.py            Tree extraction and OBB-based clustering
    add_conductor_instances.py  LAZ instance id on conductors (powerline match)
    find_extensions.py      Conductor extension detection
    find_supports.py        Pole–building support detection
    geom_utils.py           LAZ loading, DBSCAN-OBB, geometry helpers

frontend/                 Three.js web viewer
  index.html              Application shell and UI layout
  js/
    main.js               Tile loading and initialization
    viewer.js             Three.js scene, camera, rendering
    interaction.js        Selection, grouping, relation editing, shortcuts
    api.js                Backend API client
  css/style.css           Styling

scripts/                  Standalone utilities
  count_edges.py          CSV report of edge types across all graph files
  visualize_network.py    Matplotlib-based network visualization
  dev/                    Development checks (not part of the release pipeline)
    check_laz_instance_uniqueness.py  Verify instance id ↔ one semantic class per LAZ
    remap_laz_instance_ids.py         Remap: stuff classes → instance 0; others → 1..N

data/                     Runtime data (not tracked in git)
  network/                Raw network JSON files (poles, conductors)
  graph/                  Generated graph JSON files
  geometry/               Generated geometry JSON files
```

## Semantic Taxonomy

The point cloud classification follows a 15-class taxonomy:

| ID | Class        | Graph Object Type |
|----|--------------|-------------------|
| 0  | Ground       | —                 |
| 1  | Vegetation   | —                 |
| 2  | Car          | Vehicle           |
| 3  | Powerline    | Conductor         |
| 4  | Fence        | —                 |
| 5  | Tree         | Tree              |
| 6  | Pick-up      | Vehicle           |
| 7  | Van & Truck  | Vehicle           |
| 8  | Heavy-duty   | Vehicle           |
| 9  | Utility pole | Pole              |
| 10 | Light pole   | Pole              |
| 11 | Traffic pole | Pole              |
| 12 | Habitat      | Building          |
| 13 | Complex      | Building          |
| 14 | Annex        | Building          |

## Installation

**Requirements:** Python 3.8+, a modern browser with WebGL support.

```bash
pip install -r requirements.txt
```

Dependencies: `fastapi`, `uvicorn`, `numpy`, `scipy`, `laspy`, `scikit-image`, `shapely`, `networkx`, `pydantic`, `tqdm`, `python-multipart`.

## Data Preparation

### Input: LAZ/LAS Point Clouds

Each tile is a LAZ (or LAS) file with per-point fields:

- **classification** — semantic class ID (see taxonomy above)
- **instance** (or equivalent) — per-object instance ID, **unique across all semantic classes** in the tile (dataset convention)

Dev helpers: `scripts/dev/check_laz_instance_uniqueness.py` (see project tree). To normalize instance ids, `scripts/dev/remap_laz_instance_ids.py` sets instance **0** for comma-separated **stuff** semantic classes (`--stuff-classes`, default `0,1,4`) and assigns contiguous **1…N** for all other classes.

### Input: Network JSON (Optional)

If an electrical network description is available, provide a JSON file per tile with node/link definitions (poles, conductors, connectivity). This enables the pipeline to compute extensions, bifurcations, crosses, connector spans, and electrical grids.

Without a network file, the pipeline still processes buildings, vehicles, and trees.

## Usage

### Option A: Batch Pipeline (CLI)

Process all tiles from a network directory and LAZ directory:

```bash
export PYTHONPATH=$PYTHONPATH:.
python pipeline/build_full_graph.py \
  --input data/network \
  --laz_dir path/to/laz_tiles \
  --output data/graph \
  --geom_output data/geometry
```

This produces `graph_<tile_id>.json` and `geom_<tile_id>.json` for each tile.

### Option B: Direct LAZ Upload (UI)

Start the server and use the built-in upload feature:

```bash
python -m backend.main
```

Open `http://localhost:8000`, click **Upload**, select a LAZ/LAS file (and optionally a network JSON), and click **Process**. The tile is processed on-the-fly and immediately available in the viewer.

## Graph JSON Schema

Graphs are stored as **two files per tile** (and the same pattern for edits: `graph_<tile>_edit_<n>_edges.json`):

| File | Contents |
|------|-----------|
| `graph_<tile_id>.json` | Nodes (`poles`, `conductors`, `buildings`, `vehicles`, `trees`), **`building_groups`**, **`vehicle_groups`**, **`tree_groups`**, **`connector_spans`**, **`electrical_grids`**, etc. **No** top-level **`edges`** key. |
| `graph_<tile_id>_edges.json` | **Only** `{ "edges": [ ... ] }` — all relations (extension, support, adjacent, …). |

`backend.core.graph_io.load_merged_graph` / `save_split_graph` merge and split for tools. Legacy single-file graphs with an **`edges`** key inside the node file are still read correctly if the `*_edges.json` file is missing.

**`graph_<tile_id>.json`** (nodes / groups — example):

```json
{
  "poles": [
    { "id": 501, "instance_id": 501, "sem_class": 9, "is_ground": false, "is_building_support": false }
  ],
  "conductors": [
    { "link_idx": 0, "conductor_id": 0, "poles": [501, 502], "component": 0, "sem_class": 3, "instance_id": 9001 }
  ],

  "buildings": [
    { "id": 42, "sem_class": 12, "group_id": 0 }
  ],
  "vehicles": [
    { "id": 7, "sem_class": 2, "group_id": 0 }
  ],
  "trees": [
    { "id": 3 }
  ],

  "building_groups": [ { "id": 0, "members": [0, 1, 2] } ],
  "vehicle_groups":  [ { "id": 0, "members": [0, 1] } ],
  "tree_groups": [],

  "connector_spans": [
    { "id": 0, "poles": [501, 502], "conductor_ids": ["9001", "9002"], "label": "Span 501-502" }
  ],
  "electrical_grids": [
    { "id": 0, "span_ids": [0, 1, 2], "label": "Grid (3 spans)" }
  ]
}
```

**`graph_<tile_id>_edges.json`**:

```json
{
  "edges": [
    { "id": 0, "a_type": "conductor", "a_id": 9001, "b_type": "conductor", "b_id": 9002, "class": "extension" },
    { "id": 1, "a_type": "conductor", "a_id": 9001, "b_type": "pole", "b_id": 501, "class": "support" },
    { "id": 2, "a_type": "building", "a_id": 40, "b_type": "building", "b_id": 42, "class": "adjacent" },
    { "id": 3, "a_type": "building", "a_id": 42, "b_type": "tree", "b_id": 3, "class": "near" }
  ]
}
```

### `edges` (single relation list)

Every link uses **`{ "id", "a_type", "b_type", "a_id", "b_id", "class" }`**. Types include `building`, `conductor`, `pole`, `tree`, `vehicle`. Classes include **`extension`**, **`bifurcation`**, **`cross`**, **`support`** (conductor–pole), **`support_building`** (conductor–building), **`adjacent`**, **`near`**. Endpoints are stored in **canonical order** (type order then id). After `rewire_graph_to_laz_instance_ids`, **`a_id` / `b_id` for `conductor`** are **`instance_id`** values (LAZ when matched, else a synthetic id). **`pole`** endpoints use LAZ instance ids when matched.

The HTTP API and viewer receive a derived **`group_relations`** array (member_type, group_id, optional peer_type) built from proximity edges for the UI.

### Relation Types

**Building / vehicle clusters (same-type pairs)** — `a_type` and `b_type` match; `a_id` / `b_id` sorted.

- **`adjacent`** — OBB gap ≤ class-specific threshold (building: 1.0 m, vehicle: 0.25 m).
- **`near`** — Delaunay edge on cluster centroids with OBB gap ≤ threshold (building: 20.0 m, vehicle: 5.0 m). `adjacent` takes priority over `near`.

**Tree ↔ other objects** — stored as edges with types `tree` and peer type (`building`, `vehicle`, `pole`, `conductor`). No tree–tree relations; **`tree_groups` is always empty** and trees have **no** `group_id`. Classification is **gap-only** between tree footprint and peer geometry: `adjacent` if gap ≤ 0.1 m, else `near` if gap ≤ 5.0 m (`ADJACENT_DIST_MAX` / `NEAR_DIST_MAX` for `tree` in `graph_manager.py`).

### Object IDs and LAZ traceability

For `buildings`, `vehicles`, and `trees`, the graph object **`id`** is the LAZ **`instance`** value from the point cloud (via `pipeline.lib.geom_utils.load_laz_points`). The dataset guarantees **global uniqueness** of instance IDs across semantic classes within a tile, so no extra encoding is applied. **`sem_class`** is stored on **buildings** and **vehicles**; **trees** in the graph are only `{ "id" }` (semantic class for trees is fixed in the LAZ pipeline).

For **`conductors`**, **`link_idx`** and **`conductor_id`** stay as in the network; geometry keys remain **`"{link_idx}_{conductor_id}"`**. During graph generation a temporary numeric **`uid`** links edges until **`rewire_graph_to_laz_instance_ids`**: it assigns **`instance_id`** from **powerline** points when possible (`pipeline.lib.add_conductor_instances`), otherwise a **synthetic** id (`SYNTHETIC_CONDUCTOR_INSTANCE_ID_BASE + uid` in `backend.core.graph_edges`), rewrites **`edges`**, then **removes `uid`** from saved graphs. **`poles`** get **`instance_id`** from pole-class points (classes 9–11, `pipeline.lib.add_pole_instances`); **`poles[].id`** and **`geom_*.json` pole keys** follow LAZ instances when uniquely mappable.

The viewer scene conductor object uses only **`id`** (string form of **`instance_id`**).

Geometry dictionaries (`geom_*.json`) use `str(id)` as keys for buildings, vehicles, trees, and **poles** (LAZ instance id after rewire). **Conductors** stay keyed by **`link_idx_conductor_id`**.

### Clustering

**Buildings** and **vehicles** are clustered per-class using DBSCAN on OBB shortest distances; each cluster gets same-type `adjacent` / `near` relations via Delaunay on 2D centroids. **Trees are not clustered** (`tree_groups` is always `[]`). **Trees** get **undirected** `adjacent` / `near` relations to every building, vehicle, pole, and conductor in the tile (gap-based), not to other trees.

## Geometry JSON Schema

Each `geom_<tile_id>.json` contains visualization data only:

```json
{
  "scale": 1.0,
  "poles": { "<id>": { "X": 0.0, "Y": 0.0, "Z": 0.0, "footprint": { "min": [], "max": [], "rotation": [] } } },
  "conductors": { "<link_idx>_<conductor_id>": { "model": {}, "startpoint": [], "endpoint": [] } },
  "buildings": { "<id>": { "min": [], "max": [], "obb": { "center": [], "extents": [], "axes": [], "footprint": [] } } },
  "buildingHulls": { "<id>": { "vertices": [], "faces": [] } },
  "vehicles": { "<id>": { "min": [], "max": [], "obb": { "center": [], "extents": [], "axes": [], "footprint": [] } } },
  "vehicleHulls": { "<id>": { "vertices": [], "faces": [] } },
  "trees": { "<id>": { "X": 0.0, "Y": 0.0, "Z": 0.0, "height": 0.0, "crown_radius": 0.0, "obb": {} } }
}
```

## Editing Tool Controls

### Mouse

| Action              | Description                         |
|---------------------|-------------------------------------|
| Click               | Select object                       |
| Shift + Click       | Multi-select (add to selection)     |
| Shift + Drag        | Lasso selection                     |
| Double Click         | Clear selection / Reset view        |
| Left Drag           | Rotate camera                       |
| Right Drag          | Pan camera                          |
| Scroll              | Zoom                                |

### Keyboard

| Key            | Action                                |
|----------------|---------------------------------------|
| W              | Wedge (merge selected macro-instances)|
| X              | Split macro-instance                  |
| C              | Delete macro-instance (dissolve)      |
| G              | Cycle relation type (adjacent / near) |
| Space          | Cycle color scheme                    |
| Shift + Space  | Randomize colors                      |
| A              | Toggle conductors                     |
| Z              | Toggle poles                          |
| E              | Toggle buildings                      |
| R              | Toggle vehicles                       |
| T              | Toggle trees                          |
| S              | Save current edits                    |

### Grouping Operations

- **Wedge (W):** Merges two or more clusters into one. Existing inner relations are preserved; only cross-boundary pairs are computed.
- **Split (X):** Extracts the selected subset from its cluster into a new one. Cross-boundary relations are removed; inner relations of both parts are preserved.
- **Delete (C):** Removes selected members from their cluster. Relations involving removed members are deleted.

## API Endpoints

| Method | Path                              | Description                          |
|--------|-----------------------------------|--------------------------------------|
| GET    | `/api/tiles`                      | List available tile IDs              |
| GET    | `/api/tile_versions/{base_id}`    | List versions (original + edits)     |
| GET    | `/api/graph/{tile_id}`            | Load tile data for the viewer        |
| POST   | `/api/save/{tile_id}`             | Save edits as a new version          |
| POST   | `/api/edit/group`                 | Wedge / split / delete groups        |
| PATCH  | `/api/group_relation/{tile_id}`   | Update group relations               |
| POST   | `/api/load_laz`                   | Upload and process a LAZ/LAS file    |

## License

*To be determined.*
