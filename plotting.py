from dolfinx import fem, plot
import numpy as np
import pyvista as pv
import warnings

warnings.filterwarnings("ignore")

pv.global_theme.enable_camera_orientation_widget = True
pv.set_plot_theme("paraview")
pv.start_xvfb(wait=0.1)
pv.set_jupyter_backend("panel")
pv.global_theme.font.color = "black"


def get_grid(V):
    topology, cell_types, geometry = plot.create_vtk_mesh(V)
    dim = V.mesh.topology.dim
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    return grid, dim


def plot_mesh(mesh):
    """Plot DOLFIN mesh."""

    V = fem.FunctionSpace(mesh, ("CG", 1))
    grid, dim = get_grid(V)

    # Create plotter and pyvista grid
    p = pv.Plotter()

    p.add_mesh(grid, show_edges=True)
    p.view_xy()
    p.show_axes()
    p.screenshot("mesh.png", transparent_background=True)
    if not pv.OFF_SCREEN:
        p.show()


def plot_def(u, scale=1.0, **kwargs):
    V = u.function_space
    topology, cell_types, geometry = plot.create_vtk_mesh(V)
    dim = V.mesh.topology.dim
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # Create plotter and pyvista grid
    p = pv.Plotter()

    # Attach vector values to grid and warp grid by vector
    u_3D = np.zeros((geometry.shape[0], 3))
    u_3D[:, :dim] = u.x.array.reshape((geometry.shape[0], dim))
    grid["u"] = u_3D
    p.add_mesh(grid, style="surface", color="gray", opacity=0.5)
    warped = grid.warp_by_vector("u", factor=scale)
    p.add_mesh(warped, show_edges=False, cmap="viridis")
    p.show_axes()
    if dim == 2:
        p.view_xy()
    p.screenshot("deformation.png", transparent_background=True)
    if not pv.OFF_SCREEN:
        p.show()


def interpolate_expr(expr, V, name=""):
    f = fem.Function(V, name=name)
    f_expr = fem.Expression(expr, V.element.interpolation_points())
    f.interpolate(f_expr)
    return f


def evaluate_on_points(field, points):
    """This function returns the values of a field on a set of points

    Parameters
    ==========
    field: The FEniCS function from which we want points: a n x 3 np.array
    with the coordinates of the points where to evaluate the function

    It returns:
    - points_on_proc: the local slice of the point array
    - values_on_proc: the local slice of the values
    """

    import dolfinx.geometry

    mesh = field.function_space.mesh
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    # for each point, compute a colliding cells and append to the lists
    points_on_proc = []
    cells = []
    cell_candidates = dolfinx.geometry.compute_collisions(
        bb_tree, points
    )  # get candidates
    colliding_cells = dolfinx.geometry.compute_colliding_cells(
        mesh, cell_candidates, points
    )  # get actual
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            cc = colliding_cells.links(i)[0]
            points_on_proc.append(point)
            cells.append(cc)
    # convert to numpy array
    points_on_proc = np.array(points_on_proc)
    cells = np.array(cells)
    values_on_proc = field.eval(points_on_proc, cells)
    return values_on_proc, points_on_proc


def plot_stress(stress, mesh, clim=None):
    """Plot scalar stress."""

    V_sig = fem.FunctionSpace(mesh, ("DG", 0))
    V = fem.FunctionSpace(mesh, ("CG", 1))
    sig = interpolate_expr(stress, V_sig)

    grid, dim = get_grid(V)

    # Create plotter and pyvista grid
    p = pv.Plotter()

    # Attach vector values to grid and warp grid by vector
    grid.cell_data["Stress"] = sig.vector.array
    grid.set_active_scalars("Stress")

    print(
        f"Stress: (min) {sig.vector.array.min():.4f} -- {sig.vector.array.max():.4f} (max)"
    )

    p.add_mesh(
        grid,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Stress", "interactive": True},
        clim=clim,
        cmap="bwr",
    )
    if dim == 2:
        p.view_xy()
    p.show_axes()
    p.screenshot("stresses.png", transparent_background=True)
    if not pv.OFF_SCREEN:
        p.show()


def get_over_line(expr, points, mesh):
    V = fem.FunctionSpace(mesh, ("CG", 1))
    fun = interpolate_expr(expr, V)
    return evaluate_on_points(fun, points)


# over a facet:
# geometry_entitites = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, facet_markers.find(1), False)
# points = mesh.geometry.x[np.sort(geometry_entitites.flatten()),:]


def plot_over_line(expr, line, mesh, N=500):
    points = np.zeros((N, 3))
    points[:, 0] = np.linspace(line[0][0], line[1][0], N)
    points[:, 1] = np.linspace(line[0][1], line[1][1], N)
    return get_over_line(expr, points, mesh)
