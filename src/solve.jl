function get_multithreading_parameters(prob::Union{FVMProblem, FVMSystem})
    u = prob.initial_condition
    nt = Threads.nthreads()
    if prob isa FVMProblem
        duplicated_du = DiffCache(similar(u, length(u), nt))
        dirichlet_nodes = collect(keys(get_dirichlet_nodes(prob)))
    else
        duplicated_du = DiffCache(similar(u, size(u, 1), size(u, 2), nt))
        dirichlet_nodes = ntuple(i -> collect(keys(get_dirichlet_nodes(prob, i))), _neqs(prob))
    end
    solid_triangles = collect(each_solid_triangle(prob.mesh.triangulation))
    solid_vertices = collect(DelaunayTriangulation.each_point_index(prob.mesh.triangulation)) # we check for points in the vertex inside the source contribution codes
    chunked_solid_triangles = [(range, i) for (i, range) in enumerate(index_chunks(solid_triangles; n = nt))]
    boundary_edges = collect(keys(get_boundary_edge_map(prob.mesh.triangulation)))
    chunked_boundary_edges = [(range, i) for (i, range) in enumerate(index_chunks(boundary_edges; n = nt))]
    return (
        duplicated_du = duplicated_du,
        dirichlet_nodes = dirichlet_nodes,
        solid_triangles = solid_triangles,
        solid_vertices = solid_vertices,
        chunked_solid_triangles = chunked_solid_triangles,
        boundary_edges = boundary_edges,
        chunked_boundary_edges = chunked_boundary_edges,
        parallel = Val(true),
        prob = prob,
    )
end

function get_serial_parameters(prob::Union{FVMProblem, FVMSystem})
    return (
        parallel = Val(false),
        prob = prob,
    )
end

function get_fvm_parameters(prob::Union{FVMProblem, FVMSystem}, parallel::Val{B}) where {B}
    if B
        return get_multithreading_parameters(prob)
    else
        return get_serial_parameters(prob)
    end
end

"""
    jacobian_sparsity(prob)

Returns a prototype for the Jacobian of the given `prob`.
"""
jacobian_sparsity(prob::FVMProblem) = jacobian_sparsity(prob.mesh.triangulation)
function jacobian_sparsity(prob::FVMSystem{N}) where {N}
    return jacobian_sparsity(prob.mesh.triangulation, N)
end
jacobian_sparsity(prob::SteadyFVMProblem) = jacobian_sparsity(prob.problem)

function jacobian_sparsity(tri)
    I = Int64[]   # row indices
    J = Int64[]   # col indices
    V = Float64[] # values (all 1)
    n = DelaunayTriangulation.num_solid_vertices(tri)
    sizehint!(I, 6n) # points have, on average, six neighbours in a DelaunayTriangulation
    sizehint!(J, 6n)
    sizehint!(V, 6n)
    for i in DelaunayTriangulation.each_point_index(tri)
        push!(I, i)
        push!(J, i)
        push!(V, 1.0)
        !DelaunayTriangulation.has_vertex(tri, i) && continue
        for j in get_neighbours(tri, i)
            DelaunayTriangulation.is_ghost_vertex(j) && continue
            push!(I, i)
            push!(J, j)
            push!(V, 1.0)
        end
    end
    return sparse(I, J, V)
end

# Some working:
# Suppose we have a problem that looks like this (↓ vars → node)
#   u₁¹     u₂¹     u₃¹     ⋯       uₙ¹
#   u₁²     u₂²     u₃²     ⋯       uₙ²
#    ⋮        ⋮       ⋮       ⋱       ⋮
#   u₁ᴺ     u₂ᴺ     u₃ᴺ     ⋯       uₙᴺ
# When we write down the relationships here, we need
# to use the linear subscripts, so that the problem above
# is interpreted as
#   u¹      uᴺ⁺¹     u²ᴺ⁺¹     ⋯       u⁽ⁿ⁻¹⁾ᴺ⁺¹
#   u²      uᴺ⁺²     u²ᴺ⁺²     ⋯       u⁽ⁿ⁻¹⁾ᴺ⁺²
#    ⋮        ⋮       ⋮          ⋱       ⋮
#   uᴺ      u²ᴺ      u³ᴺ       ⋯       uⁿᴺ
# With this, the ith node is at the linear indices
# (i, 1), (i, 2), …, (i, N) ↦ (i-1)*N + j for j in 1:N.
# In the original matrix, a node i being related to a node ℓ
# means that the ith and ℓ columns are all related to eachother.
function jacobian_sparsity(tri, N)
    I = Int64[]   # row indices
    J = Int64[]   # col indices
    V = Float64[] # values (all 1)
    n = DelaunayTriangulation.num_solid_vertices(tri)
    sizehint!(I, 6n * N) # points have, on average, six neighbours in a DelaunayTriangulation.
    sizehint!(J, 6n * N)
    sizehint!(V, 6n * N)
    for i in DelaunayTriangulation.each_point_index(tri)
        # First, i is related to itself, meaning
        # (i, 1), (i, 2), …, (i, N) are all related.
        for ℓ in 1:N
            node = (i - 1) * N + ℓ
            for j in 1:N
                node2 = (i - 1) * N + j
                push!(I, node)
                push!(J, node2)
                push!(V, 1.0)
            end
        end
        !DelaunayTriangulation.has_vertex(tri, i) && continue
        for j in get_neighbours(tri, i)
            DelaunayTriangulation.is_ghost_vertex(j) && continue
            for ℓ in 1:N
                node = (i - 1) * N + ℓ
                for k in 1:N
                    node2 = (j - 1) * N + k
                    push!(I, node)
                    push!(J, node2)
                    push!(V, 1.0)
                end
            end
        end
    end
    return sparse(I, J, V)
end

@inline function dirichlet_callback(f::F, has_saveat, has_dir) where {F}
    if has_dir
        cb = DiscreteCallback(
            (u, t, integrator) -> true,
            f,
            save_positions = (!has_saveat, !has_saveat)
        )
    else
        cb = CallbackSet()
    end
    return cb
end

"""
    get_dirichlet_callback(prob[, f=update_dirichlet_nodes!]; kwargs...)

Get the callback for updating [`Dirichlet`](@ref) nodes. The `kwargs...` argument is ignored,
except to detect if a user has already provided a callback, in which case the
callback gets merged into a `CallbackSet` with the [`Dirichlet`](@ref) callback. If the problem
`prob` has no [`Dirichlet`](@ref) nodes, the returned callback does nothing and is never
called.

You can provide `f` to change the function that updates the [`Dirichlet`](@ref) nodes.
"""
@inline function get_dirichlet_callback(
        prob, f::F = update_dirichlet_nodes!; saveat = (),
        callback = CallbackSet(), kwargs...
    ) where {F}
    has_dir_nodes = has_dirichlet_nodes(prob)
    dir_callback = dirichlet_callback(f, !isempty(saveat), has_dir_nodes)
    cb = CallbackSet(dir_callback, callback)
    return cb
end

function SciMLBase.ODEProblem(
        prob::Union{FVMProblem, FVMSystem};
        specialization::Type{S} = SciMLBase.AutoSpecialize,
        jac_prototype = jacobian_sparsity(prob),
        parallel::Val{B} = Val(true),
        kwargs...
    ) where {S, B}
    initial_time = prob.initial_time
    final_time = prob.final_time
    time_span = (initial_time, final_time)
    initial_condition = prob.initial_condition
    cb = get_dirichlet_callback(prob; kwargs...)
    f = ODEFunction{true, S}(fvm_eqs!; jac_prototype)
    p = get_fvm_parameters(prob, parallel)
    ode_problem = ODEProblem{true, S}(f, initial_condition, time_span, p; callback = cb)
    return ode_problem
end

function SciMLBase.SteadyStateProblem(
        prob::SteadyFVMProblem;
        specialization::Type{S} = SciMLBase.AutoSpecialize,
        jac_prototype = jacobian_sparsity(prob),
        parallel::Val{B} = Val(true),
        kwargs...
    ) where {S, B}
    ode_prob = ODEProblem(prob.problem; specialization, jac_prototype, parallel, kwargs...)
    nl_prob = SteadyStateProblem{true}(ode_prob.f, ode_prob.u0, ode_prob.p; ode_prob.kwargs...)
    return nl_prob
end

function CommonSolve.init(
        prob::Union{FVMProblem, FVMSystem}, args...;
        specialization::Type{S} = SciMLBase.AutoSpecialize,
        jac_prototype = jacobian_sparsity(prob),
        parallel::Val{B} = Val(true),
        kwargs...
    ) where {S, B}
    ode_prob = SciMLBase.ODEProblem(prob; specialization, jac_prototype, parallel, kwargs...)
    return CommonSolve.init(ode_prob, args...; kwargs...)
end

"""
    solve(prob::Union{FVMProblem, FVMSystem}, args...;
        specialization=SciMLBase.AutoSpecialize,
        jac_prototype=jacobian_sparsity(prob),
        parallel=Val(true), kwargs...)

Solve the time-dependent finite-volume problem `prob` with a compatible SciML solver.

!!! warning "Missing vertices"

    When the underlying triangulation has points that are not vertices, the solver does
    not update their solution values; they remain at their initial values.

# Arguments

- `prob`: An [`FVMProblem`](@ref) or [`FVMSystem`](@ref).
- `args...`: Arguments forwarded to `solve` for the generated `ODEProblem`, normally
  including the solver algorithm.

# Keyword Arguments

- `specialization=SciMLBase.AutoSpecialize`: Controls SciML function specialization.
- `jac_prototype=jacobian_sparsity(prob)`: The Jacobian sparsity prototype.
- `parallel=Val(true)`: Set to `Val(false)` to assemble equations serially.
- `kwargs...`: Forwarded to the generated `ODEProblem` and solver.

# Returns

The solver's SciML solution. For an `FVMProblem`, each solution component corresponds to
a mesh node. For an `FVMSystem`, element `(j, i)` corresponds to variable `j` at node `i`.
"""
function CommonSolve.solve(
        prob::Union{FVMProblem, FVMSystem}, args...;
        specialization::Type{S} = SciMLBase.AutoSpecialize,
        jac_prototype = jacobian_sparsity(prob),
        parallel::Val{B} = Val(true),
        kwargs...
    ) where {S, B}
    ode_prob = SciMLBase.ODEProblem(prob; specialization, jac_prototype, parallel, kwargs...)
    return CommonSolve.solve(ode_prob, args...; kwargs...)
end

"""
    solve(prob::SteadyFVMProblem, args...;
        specialization=SciMLBase.AutoSpecialize,
        jac_prototype=jacobian_sparsity(prob),
        parallel=Val(true), kwargs...)

Solve the steady finite-volume problem `prob` with a compatible nonlinear solver.

# Arguments

- `prob`: A [`SteadyFVMProblem`](@ref).
- `args...`: Arguments forwarded to `solve` for the generated `SteadyStateProblem`,
  normally including the solver algorithm.

# Keyword Arguments

- `specialization=SciMLBase.AutoSpecialize`: Controls SciML function specialization.
- `jac_prototype=jacobian_sparsity(prob)`: The Jacobian sparsity prototype.
- `parallel=Val(true)`: Set to `Val(false)` to assemble equations serially.
- `kwargs...`: Forwarded to the generated `SteadyStateProblem` and solver.

# Returns

The nonlinear solver's SciML solution. Indexing follows the underlying `FVMProblem` or
`FVMSystem` as described for the time-dependent `solve` method.
"""
function CommonSolve.solve(
        prob::SteadyFVMProblem, args...;
        specialization::Type{S} = SciMLBase.AutoSpecialize,
        jac_prototype = jacobian_sparsity(prob),
        parallel::Val{B} = Val(true),
        kwargs...
    ) where {S, B}
    nl_prob = SciMLBase.SteadyStateProblem(
        prob; specialization, jac_prototype, parallel, kwargs...
    )
    return CommonSolve.solve(nl_prob, args...; kwargs...)
end
