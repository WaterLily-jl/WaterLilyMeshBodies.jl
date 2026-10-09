# Signed distance field and measure functions

using StaticArrays
using WaterLily
import WaterLily: @loop, δ, loc, derivative, jacobian

# measure d,n,V
function WaterLily.measure(body::MeshBody{T},x::AbstractVector{T},t;fastd²=Inf) where T
    # locate the closest point on the mesh
    ξ,(;index,d²,n,p) = locate(x, t,body,T(fastd²))
    index==0 && return (T(√fastd²),zero(x),zero(x)) # no triangles within init_d²
    # signed Euclidian distance
    d = body.boundary ? copysign(√d²,n'*(ξ-p)) : √d² - body.half_thk
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    # velocities
    v = get_velocity(p, body.mesh[index], body.velocity[index])
    dξdt = derivative(t->body.map(x,t), t)
    # x-form back with Jacobian
    dξdx = jacobian(x->body.map(x,t), x)
    return (d,hat(dξdx'n),dξdx\(v-dξdt))
end

# measure d only
@inline function WaterLily.sdf(body::MeshBody{T},x::AbstractVector{T},t;fastd²=1) where T
    ξ,(;index,d²,n,p) = locate(x, t,body,T(fastd²))
    index==0 && return T(√fastd²) # no triangles within init_d²
    body.boundary ? copysign(√d²,n'*(ξ-p)) : √d² - body.half_thk
end
@inline function locate(x,t,body,fastd²)
    ξ = body.map(x, t)
    ξ,closest(ξ, body.bvh, body.mesh; init_d²=body.boundary ? fastd² : fastd² + body.half_thk^2)
end
"""
    measure_sdf!(a::AbstractArray, body::MeshBody, t=0; fastd²=1)

Fill `a` with the signed distance from `body` at time `t`. The distance is computed exactly within `d² ≤ fastd²`,
and set to `±√fastd²` outside this region. The method depends on `body.boundary`:
 - `body.boundary == true`: The sign of the distance is determined by a flood-fill from scratch. This requires `body.mesh` to be a closed manifold.
   Wrap the body in a `NarrowBand` to only measure it near its surface, and warm-start the flood-fill.
 - `body.boundary == false`: The mesh is treated as a thin shell with half-thickness `body.half_thk`.
"""
function WaterLily.measure_sdf!(d::AbstractArray{T}, body::MeshBody{T}, t=zero(T); fastd²=1) where T
    body.boundary && return measure_sdf!(d, NarrowBand(body, size(d).-2; mem=Base.typename(typeof(d)).wrapper), t; fastd²)
    @inside d[I] = sdf(body, loc(0,I,T), t; fastd²)
end

# Only seed the NarrowBand flood-fill outside the BVH: the far-field distance of a closed MeshBody is unsigned
function WaterLilyNarrowBand.outside!(reached, body::MeshBody, t)
    body.boundary || return
    bvh, map = body.bvh, x->body.map(x,t)
    @loop reached[I] = reached[I] && dist(map(loc(0,I)), bvh.nodes[1])>1 over I ∈ CartesianIndices(reached)
end
