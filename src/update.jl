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
update!(body::AbstractBody,args...) = body
update!(body::SetBody,args...) = SetBody(body.op,update!(body.a,args...),update!(body.b,args...))
