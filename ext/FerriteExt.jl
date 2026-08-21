module FerriteExt

using Ferrite
using GeometryBasics
import WaterLilyMeshBodies: MeshBody, SetBody, save!, update!,
                            wetfacets, facetnodes, wetfaces, wetentities, facet_weights, facet_loads

# cells whose reference shape is a surface (shells) or a volume (solids)
const SurfaceCell = Ferrite.AbstractCell{<:Ferrite.AbstractRefShape{2}}
const VolumeCell  = Ferrite.AbstractCell{<:Ferrite.AbstractRefShape{3}}

# get nodes from the mesh, these are always the same
GeometryBasics.coordinates(grid::Grid{3,P,T}) where {P,T} = Point{3, T}[n.x.data for n in grid.nodes]

# convert Ferrite's native cell types into GeometryBasics faces, which are always flat and have 3 or 4 vertices
subfaces(c::Ferrite.Quadrilateral) = (c.nodes,)
# Q8 (serendipity)
subfaces(c::Ferrite.SerendipityQuadraticQuadrilateral) = (c.nodes[1:4],)
# Q9 split into 4 bilinear sub-quads
function subfaces(c::Ferrite.QuadraticQuadrilateral)
    n = c.nodes
    return ((n[1], n[5], n[9], n[8]),
            (n[5], n[2], n[6], n[9]),
            (n[9], n[6], n[3], n[7]),
            (n[8], n[9], n[7], n[4]))
end
# S3, easy
subfaces(c::Ferrite.Triangle) = (c.nodes,)
# S6 split into 4 sub-triangles
function subfaces(c::Ferrite.QuadraticTriangle)
    n = c.nodes
    return ((n[1], n[4], n[6]),
            (n[4], n[2], n[5]),
            (n[6], n[5], n[3]),
            (n[4], n[5], n[6]))
end

# decompose a Ferrite shell grid into a GeometryBasics mesh, which is what MeshBody expects
function GeometryBasics.decompose(::Type{F}, grid::Grid{3,P,T}) where {P<:SurfaceCell,T,F<:AbstractFace}
    faces = F[]
    for c in grid.cells
        for f in subfaces(c) # in cases we have to decompose the mesh
            push!(faces, F(f))
        end
    end
    return faces
end

# do we have a tri or a quad?
facetype(c) = length(Ferrite.vertices(c)) == 3 ? TriangleFace{Int} : QuadFace{Int}

"""
    wetfacets(grid::Grid, top=ExclusiveTopology(grid))

The `FacetIndex`s of the wet (outer) surface of a volume `grid`, that is the facets that are
not shared with a neighbouring cell. Pass `top` to avoid rebuilding the topology.
"""
function wetfacets(grid::Grid, top=ExclusiveTopology(grid))
    neighborhood = Ferrite.get_facet_facet_neighborhood(top, grid)
    wet = FacetIndex[]
    for (idx,neighbors) in pairs(neighborhood) # idx::CartesianIndex(cell_id, local_facet_id)
        isempty(neighbors) || continue # shared with another cell, so it is dry
        cell_id,facet_id = idx[1],idx[2]
        # the neighborhood is padded to the cell with the most facets on mixed grids
        facet_id > nfacets(getcells(grid, cell_id)) && continue
        push!(wet, FacetIndex(cell_id, facet_id))
    end
    return wet
end

"""
    facetnodes(grid::Grid, facet::FacetIndex)

The global node ids of `facet`, ordered `(corners..., mid-edges..., center)` and wound such
that the facet normal points out of the cell.
"""
function facetnodes(grid::Grid, facet::FacetIndex)
    cell = getcells(grid, facet[1])
    nodes = Ferrite.get_node_ids(cell)
    # the geometric interpolation carries the higher-order nodes that `facets(cell)` drops
    local_ids = Ferrite.facetdof_indices(geometric_interpolation(typeof(cell)))[facet[2]]
    return map(i->nodes[i], local_ids)
end

# split a facet's nodes into triangles, mirroring `subfaces` but keyed on the facet, not the
# cell, so that a mixed tri/quad surface still gives a concretely typed vector of faces
tri(a,b,c) = TriangleFace{Int}(a,b,c)
subfacets(n::NTuple{3}) = (tri(n...),)                               # linear tri
subfacets(n::NTuple{4}) = (tri(n[1],n[2],n[3]), tri(n[1],n[3],n[4])) # linear quad
subfacets(n::NTuple{6}) = (tri(n[1],n[4],n[6]), tri(n[4],n[2],n[5]), # T6
                           tri(n[6],n[5],n[3]), tri(n[4],n[5],n[6]))
subfacets(n::NTuple{8}) = subfacets(n[1:4])                          # Q8, no center node
subfacets(n::NTuple{9}) = (subfacets((n[1],n[5],n[9],n[8]))..., subfacets((n[5],n[2],n[6],n[9]))...,
                           subfacets((n[9],n[6],n[3],n[7]))..., subfacets((n[8],n[9],n[7],n[4]))...)

"""
    wetentities(grid::Grid, args...)

The global node ids of each wet entity of `grid`: every cell of a shell grid is wet, only the
outer facets of a volume grid are (see [`wetfacets`](@ref), whose `top` is forwarded). These
are the entities the surface loads are integrated over, see [`facet_loads`](@ref).
"""
wetentities(grid::Grid{3,P}) where P<:SurfaceCell = [Ferrite.get_node_ids(c) for c in grid.cells]
wetentities(grid::Grid{3,P}, args...) where P<:VolumeCell =
    [facetnodes(grid, facet) for facet in wetfacets(grid, args...)]

"""
    wetfaces(grid::Grid, args...)

The triangulation of the wet surface of `grid` as global Ferrite node ids, in the same order
as the triangles of `MeshBody(grid).mesh`, ie. each wet entity split by `subfacets`.

Use it to push a Ferrite nodal solution onto the body, see `update!(::MeshBody,faces,x,dt)`.
"""
function wetfaces(grid::Grid, args...)
    faces = TriangleFace{Int}[]
    for nodes in wetentities(grid, args...)
        append!(faces, subfacets(nodes))
    end
    return faces
end

# the interpolation of a wet entity, keyed on its number of nodes: 3,6 are triangles and
# 4,8,9 quadrilaterals, which is exactly the set `subfacets` splits
facet_interpolation(::Val{3}) = Ferrite.Lagrange{Ferrite.RefTriangle,1}()
facet_interpolation(::Val{6}) = Ferrite.Lagrange{Ferrite.RefTriangle,2}()
facet_interpolation(::Val{4}) = Ferrite.Lagrange{Ferrite.RefQuadrilateral,1}()
facet_interpolation(::Val{8}) = Ferrite.Serendipity{Ferrite.RefQuadrilateral,2}()
facet_interpolation(::Val{9}) = Ferrite.Lagrange{Ferrite.RefQuadrilateral,2}()

"""
    facet_weights(N::Int)

`W[a,k]`, the share of the force on the `k`th `subfacets` triangle that a constant traction
puts on node `a` of an `N`-node wet entity,

    W[a,k] = ∫_Tₖ Nₐ dΓ / |Tₖ|

so that `f[a] = Σₖ W[a,k]*F[k]` is the consistent nodal load of the per-triangle forces `F`.
Exact for straight-sided entities, where the sub-triangle areas cancel out of the integral.

The columns sum to one, so the resultant force is preserved whatever the weights. Weights can
be negative: a constant traction puts `-1/12` of the load on each corner of a Q8 facet.
"""
function facet_weights(N::Int)
    ip = facet_interpolation(Val(N)); X = Ferrite.reference_coordinates(ip)
    qr = Ferrite.QuadratureRule{Ferrite.RefTriangle}(4) # exact for Nₐ on a sub-triangle
    pts = Ferrite.getpoints(qr); wts = Ferrite.getweights(qr)./sum(Ferrite.getweights(qr))
    triangles = subfacets(ntuple(identity,N)); W = zeros(N, length(triangles))
    for (k,t) in enumerate(triangles), (q,p) in enumerate(pts)
        ξ = (1-p[1]-p[2])*X[t[1]] + p[1]*X[t[2]] + p[2]*X[t[3]] # sub-triangle -> entity
        for a in 1:N
            W[a,k] += wts[q]*Ferrite.reference_shape_value(ip, ξ, a)
        end
    end
    return W
end

"""
    facet_loads(grid::Grid, F::AbstractMatrix, entities=wetentities(grid))

The consistent nodal loads on `grid` of the surface forces `F`, an `Ntriangles×3` array of the
force on each triangle of `wetfaces(grid)`, eg. `Array(sf.pressure).+Array(sf.viscous)` of a
`SurfaceForces`. Returns a `3×Nnodes` array, `f[:,n]` being the load on node `n`.

Hoist `entities` out of the time loop, it rebuilds the grid topology on every call.
```julia
entities = WaterLilyMeshBodies.wetentities(grid)  # once
f = WaterLilyMeshBodies.facet_loads(grid, F, entities)
```
This integrates the traction against the shape functions of the wet entity, which the naive
`F/3` scatter over the triangle vertices only reproduces for a linear triangle. The traction
is still piecewise constant over the triangles, which is the remaining error.
"""
function facet_loads(grid::Grid, F::AbstractMatrix{T}, entities=wetentities(grid)) where T
    W = Dict(N => facet_weights(N) for N in unique(length.(entities)))
    ntri = sum(nodes->size(W[length(nodes)],2), entities; init=0)
    @assert ntri==size(F,1) "$(size(F,1)) forces given for $ntri wet triangles"
    @assert size(F,2)==3 "the forces must be a Ntriangles×3 array"
    f = zeros(T, 3, Ferrite.getnnodes(grid)); k = 0
    for nodes in entities
        w = W[length(nodes)]
        for j in axes(w,2)
            k += 1
            for (a,n) in enumerate(nodes), d in 1:3
                f[d,n] += w[a,j]*F[k,d]
            end
        end
    end
    return f
end

# convert a Ferrite grid into a MeshBody. The faces index into the full node vector, so the
# global Ferrite node ids are preserved and `wetfaces` can be used to update the body
function MeshBody(grid::Grid{3,P}; kwargs...) where P<:Union{SurfaceCell,VolumeCell}
    points = GeometryBasics.decompose(Point{3, Float32}, grid)
    MeshBody(GeometryBasics.Mesh(points, wetfaces(grid)); kwargs...)
end

# mixed-dimension grids have no single notion of a wet surface
function MeshBody(grid::Grid{3}; kwargs...)
    throw(ArgumentError("cannot build a MeshBody from a grid of $(eltype(grid.cells)) cells, \
                         the cells must all be surface (shell) or all be volume cells"))
end

end # module
