import streamlit as st
# streamlit_app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import trimesh
import plotly.graph_objects as go

st.set_page_config(page_title="Cast-Iron DED Planner Viewer", layout="wide")

st.title("Cast-Iron DED Planner — Geometry, Toolpath, Preheat & Risk Viewer")

st.markdown(
    """
Upload a **mesh** (STL/OBJ/PLY) and optionally a **toolpath CSV**.
Toolpath CSV expected columns: `x, y, z` (optional: `power, speed, feed, dwell`).
"""
)

# -----------------------------
# Helpers
# -----------------------------
def _safe_cleanup(mesh):
    """Cleanup across trimesh versions safely."""
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "merge_vertices"):
        try:
            mesh.merge_vertices()
        except TypeError:
            # older trimesh versions
            mesh.merge_vertices(merge_tex=False)
    return mesh


@st.cache_data(show_spinner=False)
def load_mesh_bytes(file_bytes: bytes, filename: str):
    """Load mesh from bytes (cached)."""
    if not file_bytes or not filename:
        return None

    ext = filename.split(".")[-1].lower()
    mesh = trimesh.load(io.BytesIO(file_bytes), file_type=ext, force="mesh")

    if mesh is None:
        return None

    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            return None
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    if getattr(mesh, "is_empty", True):
        return None

    mesh = _safe_cleanup(mesh)
    return mesh


def parse_toolpath(file_bytes: bytes, filename: str):
    """Load toolpath CSV -> dataframe with lowercase cols."""
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"x", "y", "z"}
    if not required.issubset(set(df.columns)):
        raise ValueError("Toolpath CSV must contain columns: x, y, z")
    return df


def bbox_center_and_size(mesh: trimesh.Trimesh):
    bounds = np.asarray(mesh.bounds)  # [[minx,miny,minz],[maxx,maxy,maxz]]
    center = (bounds[0] + bounds[1]) / 2.0
    size = (bounds[1] - bounds[0])
    return center, size, bounds


def transform_toolpath_to_mesh(tp_xyz: np.ndarray, mesh: trimesh.Trimesh, mode: str):
    """
    mode:
      - "none": no change
      - "center": translate to mesh centroid (match centers)
      - "fit_bbox": scale+translate tp bbox to mesh bbox
    """
    if mode == "none" or mesh is None:
        return tp_xyz

    center, size, bounds = bbox_center_and_size(mesh)

    tp = tp_xyz.copy()
    tp_center = tp.mean(axis=0)

    if mode == "center":
        tp = tp - tp_center + center
        return tp

    if mode == "fit_bbox":
        tp_min = tp.min(axis=0)
        tp_max = tp.max(axis=0)
        tp_size = np.maximum(tp_max - tp_min, 1e-9)

        # scale to fit within mesh bbox (uniform scale)
        s = float(np.min(size / tp_size))
        tp = (tp - tp_center) * s + center
        return tp

    return tp


def make_mesh_trace(mesh: trimesh.Trimesh, opacity: float):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        opacity=float(opacity),
        name="Part Mesh"
    )


def make_toolpath_trace(tp: np.ndarray, z_for_color: np.ndarray, width: int):
    # Color by Z using a colorscale (no hardcoded color)
    return go.Scatter3d(
        x=tp[:, 0], y=tp[:, 1], z=tp[:, 2],
        mode="lines+markers",
        line=dict(width=int(width)),
        marker=dict(size=2, color=z_for_color, colorscale="Viridis", showscale=True),
        name="Toolpath"
    )


def approx_path_length(xyz: np.ndarray) -> float:
    diffs = np.diff(xyz, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def detect_layers(z_vals: np.ndarray, tol: float):
    """Group points by layer based on z clustering using tolerance."""
    z_sorted = np.sort(np.unique(z_vals))
    if len(z_sorted) == 0:
        return []

    layers = []
    current = [z_sorted[0]]
    for z in z_sorted[1:]:
        if abs(z - current[-1]) <= tol:
            current.append(z)
        else:
            layers.append(float(np.mean(current)))
            current = [z]
    layers.append(float(np.mean(current)))
    return layers


def layer_mask(tp: np.ndarray, layer_values: list[float], upto_idx: int, tol: float):
    """Mask points whose z <= layer_values[upto_idx] (within tol)."""
    if not layer_values:
        return np.ones(len(tp), dtype=bool)
    zmax = layer_values[min(upto_idx, len(layer_values) - 1)]
    return tp[:, 2] <= (zmax + tol)


def generate_example_raster_csv(nx=5, ny=3, layers=3, step=5.0, layer_h=1.0):
    rows = []
    rows.append(["x", "y", "z"])
    for k in range(layers):
        z = k * layer_h
        for j in range(ny):
            y = j * step
            xs = [i * step for i in range(nx)]
            if j % 2 == 1:
                xs = list(reversed(xs))
            for x in xs:
                rows.append([x, y, z])
    out = io.StringIO()
    for r in rows:
        out.write(",".join(map(str, r)) + "\n")
    return out.getvalue().encode("utf-8")


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Viewer Controls")
mesh_opacity = st.sidebar.slider("Mesh opacity", 0.1, 1.0, 0.6, 0.05)
show_vertices = st.sidebar.checkbox("Show mesh vertices (debug)", value=False)
path_width = st.sidebar.slider("Toolpath line width", 1, 12, 6, 1)

st.sidebar.divider()
st.sidebar.header("Toolpath Alignment")
align_mode = st.sidebar.selectbox(
    "Align toolpath to mesh",
    ["none", "center", "fit_bbox"],
    index=1,
    help="Use 'center' if your path is in a different origin; 'fit_bbox' scales+centers a test path into the part."
)

st.sidebar.divider()
st.sidebar.header("Layer Controls")
layer_tol = st.sidebar.number_input("Layer detection tolerance (Z)", min_value=0.0, value=0.01, step=0.01)

st.sidebar.divider()
st.sidebar.header("Preheat & Interpass (Planner Inputs)")
preheat_c = st.sidebar.slider("Preheat target (°C)", 20, 600, 315, 5)
ramp_c_per_min = st.sidebar.slider("Heating ramp rate (°C/min)", 1, 50, 10, 1)
interpass_margin_c = st.sidebar.slider("Interpass margin (°C above preheat)", 0, 150, 40, 5)
interpass_min_c = preheat_c + interpass_margin_c

st.sidebar.caption(f"Interpass minimum target: **{interpass_min_c} °C**")

st.sidebar.divider()
st.sidebar.header("Quick Test")
st.sidebar.caption("Generate and download an example toolpath CSV.")
ex_nx = st.sidebar.slider("Example raster X points", 3, 30, 5, 1)
ex_ny = st.sidebar.slider("Example raster Y lines", 2, 30, 3, 1)
ex_layers = st.sidebar.slider("Example layers", 1, 20, 3, 1)
ex_step = st.sidebar.slider("Example step (mm)", 1.0, 20.0, 5.0, 0.5)
ex_h = st.sidebar.slider("Layer height (mm)", 0.2, 5.0, 1.0, 0.1)

example_bytes = generate_example_raster_csv(ex_nx, ex_ny, ex_layers, ex_step, ex_h)
st.sidebar.download_button(
    label="Download example toolpath CSV",
    data=example_bytes,
    file_name="toolpath_example.csv",
    mime="text/csv"
)

# -----------------------------
# Uploads
# -----------------------------
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("1) Upload Part Mesh")
    mesh_file = st.file_uploader("Mesh file (STL/OBJ/PLY)", type=["stl", "obj", "ply"])

with c2:
    st.subheader("2) Upload Toolpath")
    path_file = st.file_uploader("Toolpath CSV (x,y,z)", type=["csv"])

mesh = None
if mesh_file is not None:
    mesh = load_mesh_bytes(mesh_file.getvalue(), mesh_file.name)

toolpath_df = None
tp_xyz = None
tp_xyz_aligned = None
layers = []
if path_file is not None:
    try:
        toolpath_df = parse_toolpath(path_file.getvalue(), path_file.name)
        tp_xyz = toolpath_df[["x", "y", "z"]].to_numpy(dtype=float)
        if mesh is not None:
            tp_xyz_aligned = transform_toolpath_to_mesh(tp_xyz, mesh, align_mode)
        else:
            tp_xyz_aligned = tp_xyz.copy()

        layers = detect_layers(tp_xyz_aligned[:, 2], tol=float(layer_tol))
    except Exception as e:
        st.error(str(e))
        toolpath_df = None
        tp_xyz = None
        tp_xyz_aligned = None
        layers = []

# -----------------------------
# Layer slider
# -----------------------------
layer_idx = 0
if tp_xyz_aligned is not None and len(layers) > 0:
    layer_idx = st.slider(
        "Show toolpath up to layer",
        min_value=0,
        max_value=len(layers) - 1,
        value=len(layers) - 1,
        step=1
    )

# -----------------------------
# Visualization
# -----------------------------
st.subheader("3D Visualization")

fig = go.Figure()

if mesh is not None:
    fig.add_trace(make_mesh_trace(mesh, opacity=mesh_opacity))

    if show_vertices:
        v = np.asarray(mesh.vertices)
        fig.add_trace(go.Scatter3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            mode="markers",
            marker=dict(size=1),
            name="Vertices"
        ))

if tp_xyz_aligned is not None:
    mask = layer_mask(tp_xyz_aligned, layers, layer_idx, tol=float(layer_tol))
    tp_show = tp_xyz_aligned[mask]
    fig.add_trace(make_toolpath_trace(tp_show, tp_show[:, 2], width=path_width))

fig.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data"
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Quick stats & “planner-like” checks (lightweight)
# -----------------------------
st.subheader("Quick Stats & Checks")
a, b, c = st.columns(3)

with a:
    if mesh is None:
        st.info("Upload a mesh to see geometry stats.")
    else:
        center, size, bounds = bbox_center_and_size(mesh)
        st.metric("Vertices", int(len(mesh.vertices)))
        st.metric("Faces", int(len(mesh.faces)))
        st.write("Mesh bbox (min → max):")
        st.code(f"{bounds[0].round(3)} → {bounds[1].round(3)}")

with b:
    if tp_xyz_aligned is None:
        st.info("Upload a toolpath CSV to see toolpath stats.")
    else:
        st.metric("Toolpath points", int(len(tp_xyz_aligned)))
        st.metric("Approx path length", f"{approx_path_length(tp_xyz_aligned):.2f}")
        st.metric("Detected layers", int(len(layers)))
        if len(layers) > 0:
            st.write(f"Layer Z values (sample): {layers[:min(8,len(layers))]}")

with c:
    st.write("Preheat & Interpass targets")
    st.metric("Preheat (°C)", int(preheat_c))
    st.metric("Ramp (°C/min)", int(ramp_c_per_min))
    st.metric("Interpass min (°C)", int(interpass_min_c))

    # Lightweight warning logic (placeholders until you add real thermal model)
    # These checks are intentionally conservative and transparent.
    if preheat_c < 205:
        st.warning("Preheat below 205°C (often cited as minimum for gray iron repair).")
    if interpass_min_c < preheat_c + 40:
        st.warning("Interpass margin below +40°C rule-of-thumb for tempering benefit (multi-pass).")

st.divider()

# -----------------------------
# “Next Modules” placeholders (so the app already looks like a thesis tool)
# -----------------------------
with st.expander("Thermal Monitoring (placeholder)"):
    st.write(
        "Next step: your solver will output temperature histories at monitor points (groove root, sidewall, bulk). "
        "You will plot T(t), dT/dt, max gradients, and enforce constraints."
    )
    st.write("Suggested monitor outputs:")
    st.code(
        "monitor_points:\n"
        "  root: [x,y,z]\n"
        "  sidewall: [x,y,z]\n"
        "  bulk: [x,y,z]\n"
        "outputs:\n"
        "  T(t), max_T, max_dTdt, interpass_T"
    )

with st.expander("Export (placeholder)"):
    st.write("Add export buttons for:")
    st.write("- plan.json (inputs + chosen strategy + scores)")
    st.write("- toolpath.csv (post-processed)")
    st.write("- report.pdf/html (plots + justification)")

st.caption("Tip: If the toolpath looks tiny compared to the mesh, set Toolpath Alignment → **fit_bbox** for testing.")
