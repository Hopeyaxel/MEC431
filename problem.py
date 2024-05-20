from dolfinx import fem
from ufl import sym, grad, inner, action, TrialFunction, TestFunction, Measure
import numpy as np


def integrate(expr):
    return fem.assemble_scalar(fem.form(expr))


def elastic_problem(mesh_data, sigma, Phi, order=1):
    """
    Defines and solves a linear elastic problem.

    Parameters
    ==========
        - `mesh_data`: a triple containing (mesh, cell_markers, facet_markers)
        - `sigma`: a function sigma(epsilon) returning the linear constitutive equation
        - `Phi`: a function Phi(v) returning the linear form of external loads
        - `order`: polynomial order of FE discretization. Always uses `order=1` in 3D.

    Returns
    =======
        - u: the solution displacement field
        - sig: the solution stress field
        - E_pot: the value of the potential energy
        - u_max: the value of the maximum displacement
    """

    mesh, cell_markers, facet_markers = mesh_data
    gdim = mesh.geometry.dim
    fdim = gdim - 1
    if gdim == 3 and order > 1:
        order = 1
        print(
            "Warning: Polynomial order has been chosen greater than 1 for a 3D computation.\n \
              Revert back to order=1 to reduce computation times..."
        )
    V = fem.VectorFunctionSpace(mesh, ("CG", order))

    # function space for the stress
    V_sig = fem.TensorFunctionSpace(mesh, ("DG", order - 1))
    sig = fem.Function(V_sig, name="Current_stress")

    def epsilon(v):
        return sym(grad(v))

    u_ = TrialFunction(V)
    v = TestFunction(V)

    dx = Measure("dx", domain=mesh)
    a = inner(sigma(epsilon(u_)), epsilon(v)) * dx
    L = Phi(v)

    u = fem.Function(V, name="Displacement")

    bottom_dofs = fem.locate_dofs_topological(V, fdim, facet_markers.find(1))
    end_dofs = fem.locate_dofs_topological(V, fdim, facet_markers.find(6))

    bcs = [
        fem.dirichletbc(np.zeros((gdim,)), bottom_dofs, V),
        fem.dirichletbc(np.zeros((gdim,)), end_dofs, V),  # no end_dofs if in 2D
    ]

    problem = fem.petsc.LinearProblem(
        a, L, u=u, bcs=bcs, petsc_options={"ksp_type": "cg", "pc_type": "ilu"}
    )
    problem.solve()

    E_pot = fem.assemble_scalar(fem.form(-0.5 * action(L, u)))

    sig = sigma(epsilon(u))

    U = u.vector.array.reshape((-1, gdim))
    imax = np.argmax(np.linalg.norm(U, axis=1))
    u_max = U[imax, :]
    print("Potential energy:", E_pot)
    print("Maximum displacement:", u_max)
    print("\n")
    return u, sig, E_pot, U[imax, :]
