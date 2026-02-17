import streamlit as st
import io
import numpy as np
import pandas as pd
import streamlit as st
import trimesh
import plotly.graph_objects as go


st.set_page_config(page_title="Cast-Iron DED Planner Viewer", layout="wide")

st.title("Cast-Iron DED Planner — Geometry & Toolpath Viewer (MVP)")

st.markdown(
    """
Upload a **mesh** (STL/OBJ/PLY) and optionally a **toolpath CSV**.
Toolpath CSV expected columns: `x, y, z` (optional: `power, speed, feed`).
"""
)

# ---------- Helpers ----------
def load_mesh_from_upload(uploaded_file):
def load_mesh_from_upload(uploaded_file):
    """Load mesh via trimesh from Streamlit UploadedFile."""
    if uploaded_file is None:
        return None

    data = uploaded_file.read()
    file_like = io.BytesIO(data)

    mesh = trimesh.load(
        file_like,
        file_type=uploaded_file.name.split(".")[-1],
        force="mesh"
    )

    if mesh is None or mesh.is_empty:
        return None

    # If it loads as a scene, merge geometry
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )

    # --- Safe cleanup across trimesh versions ---
    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()

    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()

    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()

    if hasattr(mesh, "merge_vertices"):
        mesh.merge_vertices()

    return mesh
    



def mesh_to_plotly(mesh: trimesh.Trimesh, opacity=0.6):
    """Convert a trimesh mesh to Plotly Mesh3d."""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)

    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        opacity=float(opacity),
        name="Part Mesh"
    )


def toolpath_to_plotly_line(df: pd.DataFrame, name="Toolpath", width=6):
    """Convert toolpath points (x,y,z) to a Plotly 3D line."""
    return go.Scatter3d(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        z=df["z"].to_numpy(),
        mode="lines+markers",
        line=dict(width=int(width)),
        marker=dict(size=2),
        name=name
    )


def center_and_scale(mesh: trimesh.Trimesh):
    """Center mesh for nicer viewing."""
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.centroid)
    return mesh


# ---------- Sidebar controls ----------
st.sidebar.header("Controls")

mesh_opacity = st.sidebar.slider("Mesh opacity", 0.1, 1.0, 0.6, 0.05)
show_edges = st.sidebar.checkbox("Show mesh edges (approx)", value=False)
path_width = st.sidebar.slider("Toolpath line width", 1, 12, 6, 1)

st.sidebar.divider()
st.sidebar.subheader("Example toolpath generator (optional)")
gen_example_path = st.sidebar.checkbox("Generate example helix toolpath (no CSV needed)", value=False)
n_points = st.sidebar.slider("Example points", 50, 1500, 400, 50)


# ---------- Main layout ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Upload Part Mesh")
    mesh_file = st.file_uploader("Mesh file (STL/OBJ/PLY)", type=["stl", "obj", "ply"])

with col2:
    st.subheader("2) Upload Toolpath")
    path_file = st.file_uploader("Toolpath CSV (x,y,z)", type=["csv"])


mesh = load_mesh_from_upload(mesh_file)

toolpath_df = None
if path_file is not None:
    toolpath_df = pd.read_csv(path_file)
    toolpath_df.columns = [c.strip().lower() for c in toolpath_df.columns]
    required = {"x", "y", "z"}
    if not required.issubset(set(toolpath_df.columns)):
        st.error("Toolpath CSV must contain columns: x, y, z")
        toolpath_df = None

# Example toolpath if requested
if gen_example_path:
    t = np.linspace(0, 10 * np.pi, n_points)
    r = 20.0
    z = np.linspace(0, 50.0, n_points)
    toolpath_df = pd.DataFrame({
        "x": r * np.cos(t),
        "y": r * np.sin(t),
        "z": z
    })

# ---------- Visualization ----------
st.subheader("3D Visualization")

fig = go.Figure()

if mesh is not None:
    mesh_c = center_and_scale(mesh)
    fig.add_trace(mesh_to_plotly(mesh_c, opacity=mesh_opacity))

    if show_edges:
        # Very lightweight "edge-ish" visualization: scatter vertices
        v = np.asarray(mesh_c.vertices)
        fig.add_trace(go.Scatter3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            mode="markers",
            marker=dict(size=1),
            name="Vertices"
        ))

if toolpath_df is not None:
    # If mesh is centered, we also center toolpath for consistency.
    # In real use, both should be in same coordinate frame already.
    tp = toolpath_df.copy()
    if mesh is not None:
        tp[["x", "y", "z"]] = tp[["x", "y", "z"]] - np.array(mesh.centroid)
    fig.add_trace(toolpath_to_plotly_line(tp, width=path_width))

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

# ---------- Quick stats ----------
st.subheader("Quick Stats")
c3, c4, c5 = st.columns(3)

with c3:
    if mesh is None:
        st.info("Upload a mesh to see stats.")
    else:
        st.metric("Vertices", int(len(mesh.vertices)))
        st.metric("Faces", int(len(mesh.faces)))

with c4:
    if toolpath_df is None:
        st.info("Upload a toolpath CSV to see stats.")
    else:
        st.metric("Toolpath points", int(len(toolpath_df)))
        # Approx path length
        pts = toolpath_df[["x", "y", "z"]].to_numpy()
        diffs = np.diff(pts, axis=0)
        length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        st.metric("Approx path length", f"{length:.2f}")

with c5:
    st.write("Next steps you can add:")
    st.write("- Preheat slider + constraints checks")
    st.write("- Plot T(t) at monitor points")
    st.write("- Color map risk on surface")
    st.write("- Export plan.json + toolpath.csv")
