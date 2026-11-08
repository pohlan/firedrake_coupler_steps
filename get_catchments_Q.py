import numpy as np
from collections import defaultdict
import firedrake as df
from plotting.loading_functions import *
from models_main.helpers import *
from collections import deque

def build_vertex_edges(cells):
    """
    Build unique edges from triangular mesh connectivity.

    cells: (n_cells, 3) array of local vertex indices.
    """
    edges = set()

    for a, b, c in cells:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))

    return np.asarray(sorted(edges), dtype=np.int64)


def calculate_flow_directions(phi, coordinates, edges, Q):
    """
    Calculate steepest-downhill flow direction at each vertex.

    phi:         hydraulic potential, shape (n_vertices,)
    coordinates: vertex coordinates, shape (n_vertices, ndim)
    edges:       unique vertex pairs, shape (n_edges, 2)

    Returns:
        downstream[i]: destination vertex of flow from vertex i.
                       downstream[i] == i indicates a sink.
        max_gradient[i]: maximum downhill gradient.
    """

    n_vertices = len(phi)

    downstream = np.arange(n_vertices)
    max_Q = np.zeros(n_vertices)

    # Process each edge in both directions.
    for ((i, j),Q_) in zip(edges,Q):

        distance = np.linalg.norm(coordinates[j] - coordinates[i])

        if distance == 0:
            continue

        sign_ij = np.sign(phi[i] - phi[j])
        sign_ji = -sign_ij

        # n_ij = coordinates[j] - coordinates[i]

        # qx_average = 0.5*(qx[i]+qx[j])
        # qy_average = 0.5*(qy[i]+qy[j])
        # qn_ij = (qx_average*n_ij[0] + qy_average*n_ij[1]) * np.sqrt(3)/3*distance
        # qn_ji = -qn_ij

        Q_ij = sign_ij*Q_ # + qn_ij
        Q_ji = sign_ji*Q_ # + qn_ji


        if (sign_ij>0) & (Q_ij> max_Q[i]):
            max_Q[i] = Q_ij
            downstream[i] = j

        if (sign_ji>0) & (Q_ji > max_Q[j]):
            max_Q[j] = Q_ji
            downstream[j] = i


        # gradient_ij = (phi[i] - phi[j]) / distance
        # gradient_ji = -gradient_ij

        # if gradient_ij > max_gradient[i]:
        #     max_gradient[i] = gradient_ij
        #     downstream[i] = j

        # if gradient_ji > max_gradient[j]:
        #     max_gradient[j] = gradient_ji
        #     downstream[j] = i

    return downstream, max_Q


def delineate_catchment(outlets, downstream):
    """
    Find all vertices draining into a specified outlet.

    Returns:
        Boolean array identifying catchment vertices.
    """

    n_vertices = len(downstream)

    # Construct reverse flow graph.
    upstream = [[] for _ in range(n_vertices)]

    for i, j in enumerate(downstream):
        if i != j:
            upstream[j].append(i)

    # Traverse upstream from outlet.
    catchment = np.zeros(n_vertices, dtype=int) -1

    for i_outlet, outlet_vertices in enumerate(outlets):
        for outlet in outlet_vertices:
            queue = deque([outlet])
            catchment[outlet] = i_outlet

            while queue:
                vertex = queue.popleft()

                for neighbour in upstream[vertex]:
                    if catchment[neighbour]==-1:
                        catchment[neighbour] = i_outlet
                        queue.append(neighbour)

    return catchment


def get_outflow_index_2D_mesh(mesh):

    # Cell -> vertex connectivity
    cells = mesh.coordinates.function_space().cell_node_list

    # Exterior facets
    facets = mesh.exterior_facets

    # Cell containing each boundary facet
    cell_ids = facets.facet_cell[:, 0]

    # Local facet number within that cell
    local_facet = facets.local_facet_dat.data_ro

    cell_facets = mesh.cell_to_facets.data_ro
    markers = cell_facets[cell_ids, local_facet, 1]

    # For triangles, local facet i is opposite vertex i
    edge_vertices = np.array([
        np.delete(cells[c], lf)
        for c, lf in zip(cell_ids, local_facet)
    ])

    # Select your outflow boundary (Gmsh physical group 1)
    outflow = markers == 1

    return cells, edge_vertices[outflow]  # vertices!

def get_Q_on_original_mesh(submesh, edge_coordinates, Q_vals):
    # get coordinates of dofs for both DGT0 and subDG0
    # crds_DGT0 = get_coordinates(mesh, "DGT", 0)
    crds_subDG0 = get_coordinates(submesh, "DG", 0)  # doesn't need to be subm apparently
    # loop through each coordinate of DGT0 and assign value to correct index of DG0 function
    Q_out = np.zeros(len(edge_coordinates))
    for ((xi,yi),f_val) in zip(crds_subDG0,Q_vals):
        i = np.argmin(np.sqrt((xi - edge_coordinates[:,0])**2 + (yi - edge_coordinates[:,1])**2))  # due to rounding error np.where(xi==crds_subDG0[:,0] ...) doesn't work
        Q_out[i] = f_val
    return Q_out



run_index = 903

## Step 1: Load model output
timeseries_path = f"parameter_runs/run_{run_index}/time_series.h5"
mesh, smesh = get_meshes(timeseries_path)
meshx, meshy = zip(*get_coordinates(mesh, "CG", 1))
us_raw, m_raw, phi_raw, h_raw, q_raw, Q_raw, S_raw, n_idx, _ = load_model_output(timeseries_path)
# n = int(365/2 * 15)
n = n_idx-1

V = df.FunctionSpace(mesh, "CG", 1)
# phi = df.Function(V)
# phi.dat.data[:] = phi_raw[:,n]
phi_values = phi_raw[:,n]
Q_values   = Q_raw[:,n]
# qx   = q_raw[::2, n]
# qy   = q_raw[1::2, n]


cells, outlet_vertex = get_outflow_index_2D_mesh(mesh)
edges  = build_vertex_edges(cells)
coordinates = mesh.coordinates.dat.data_ro
edge_coordinates = 0.5 * (coordinates[edges[:, 0]] + coordinates[edges[:, 1]])
Q = get_Q_on_original_mesh(smesh, edge_coordinates, Q_values)

# Calculate flow directions
downstream, Qs = calculate_flow_directions(phi_values, coordinates, edges, Q)
catchment = delineate_catchment(outlet_vertex, downstream)


# x_p = np.array([c[0] for c in mesh.coordinates.dat.data_ro])
# y_p = np.array([c[1] for c in mesh.coordinates.dat.data_ro])

# plt.figure()
# plt.scatter(x_p,y_p, c="black", s=1)
# plt.scatter(x_p[outlets],y_p[outlets])
# plt.savefig("outlets.jpg")

# x_p = np.array([c[0] for c in edge_coordinates])
# y_p = np.array([c[1] for c in edge_coordinates])
# fig, ax = plt.subplots()
# ax.scatter(x_p,y_p, c=Q, s=1)
# ax.set_aspect("equal")
# ax.set_xlim(-2.320e5, -1.8e5)
# ax.set_ylim(-2.54e6,-2.5e6)
# plt.savefig("outlets.jpg")


c = df.Function(V)
c.dat.data[:] = catchment

fig, axes = plt.subplots()
cl = tripcolor(c, axes=axes)
fig.colorbar(cl)
axes.set_aspect("equal")
plt.savefig("catchments.jpg")


