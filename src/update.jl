# Mesh update functions

import WaterLily: @loop, AbstractBody, SetBody, update!
using ImplicitBVH
import ImplicitBVH: BBox, BVH
import ConstructionBase: setproperties

"""
    update!(body::MeshBody{T},new_mesh::AbstractArray,dt=0;kwargs...)

Updates the mesh body position using the new mesh triangle coordinates.

    xᵢ(t+Δt) = x[i]
    vᵢ(t+Δt) = (xᵢ(t+Δt) - xᵢ(t))/dt
    where `x[i]` is the new (t+Δt) position of the control point, `vᵢ` is the velocity at that control point.

This function mutates internal fields of `MeshBody`, but must also replace your body in the simulation
```julia
sim.body = update!(sim.body, new_mesh, dt)
```
otherwise the `BVH` will not be updated correctly.
"""
function update!(a::MeshBody{T},new_mesh::AbstractArray,dt=0) where T
    Rs = CartesianIndices(a.mesh)
    # if nonzero time step, update the velocity field
    dt>0 && (@loop a.velocity[I] = (new_mesh[I]-a.mesh[I])/T(dt) over I in Rs)
    @loop a.mesh[I] = new_mesh[I] over I in Rs
    # update the BVH
    setproperties(a, bvh=BVH(ImplicitBVH.BBox{T}.(a.mesh), ImplicitBVH.BBox{T}))
end
"""
    update!(body::MeshBody{T},faces::AbstractArray,x::AbstractArray,dt=0)

Updates the mesh body position from the nodal positions `x` (a `3×Nnodes` array) gathered
through the triangle connectivity `faces` (an array of 3 node ids per triangle, as returned
by `wetfaces(grid)` for a Ferrite grid). The velocity is set as in `update!(body,mesh,dt)`.

As with `update!(body,mesh,dt)`, `x` is in the body's own frame: if the body was built with a
`scale`, pass `x` already scaled. `faces` and `x` must live in the same memory as `body.mesh`,
and `faces` must index into the *full* node array, ie. `x[:,n]` is the position of node `n`.
```julia
faces = WaterLilyMeshBodies.wetfaces(grid) |> mem  # once, the connectivity never changes
sim.body = update!(sim.body, faces, x, dt)         # each time the nodes move
```
"""
function update!(a::MeshBody{T},faces::AbstractArray,x::AbstractMatrix,dt=0) where T
    @assert length(faces)==length(a.mesh) "$(length(faces)) faces for $(length(a.mesh)) triangles"
    new_mesh = similar(a.mesh)
    @loop new_mesh[I] = gather(faces[I],x) over I in CartesianIndices(a.mesh)
    update!(a,new_mesh,dt)
end
# the columns of a triangle are its vertices, see `geometry.jl`
@inline gather(f,x::AbstractMatrix{T}) where T = SMatrix{3,3,T}(x[1,f[1]],x[2,f[1]],x[3,f[1]],
                                                                x[1,f[2]],x[2,f[2]],x[3,f[2]],
                                                                x[1,f[3]],x[2,f[3]],x[3,f[3]])

update!(body::AbstractBody,args...) = body
update!(body::SetBody,args...) = SetBody(body.op,update!(body.a,args...),update!(body.b,args...))
