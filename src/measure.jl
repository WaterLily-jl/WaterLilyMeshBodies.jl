# Signed distance field and measure functions

using StaticArrays
using ForwardDiff
using WaterLily
import WaterLily: @loop, δ, loc

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
    dξdt = ForwardDiff.derivative(t->body.map(x,t), t)
    # x-form back with Jacobian
    dξdx = ForwardDiff.jacobian(x->body.map(x,t), ξ)
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
and set to `√fastd²` outside this region. The method depends on `body.boundary`:
 - `body.boundary == true`: The sign of the distance is determined by a global flood-fill. This requires `body.mesh` to be a closed manifold.
 - `body.boundary == false`: The mesh is treated as a thin shell with half-thickness `body.half_thk`.
"""
function WaterLily.measure_sdf!(d::AbstractArray{T}, body::MeshBody{T}, t=zero(T); fastd²=1) where T
    # SDF within d²≤fastd²
    @inside d[I] = sdf(body, loc(0,I,T), t; fastd²)

    # Determine points inside the closed body.boundary using a flood fill
    if body.boundary
        if body.cache === nothing
            near, reached, farinside = similar(d, Bool), similar(d, Bool), similar(d, Bool)
            outside!(reached, body.bvh, x->body.map(x,t))
        else
            near, reached, farinside = body.cache
        end
        flood_fill!(near, reached, farinside, d)
        @inside d[I] = farinside[I] ? -abs(d[I]) : d[I]
    end
end

# Flood-fill to classify points as inside or outside a closed boundary
outside!(reached,bvh,map) = @loop reached[I] = dist(map(loc(0,I)), bvh.nodes[1])>1 over I ∈ CartesianIndices(reached)
function flood_fill!(near, reached, scratch, sdf; cutoff=1f0)
    @. near = abs(sdf) < cutoff
    copyto!(scratch,reached)
    for _ in 1:min(size(near)...)÷2
        @inside scratch[I] = flood_update(I, reached, near)
        @inside reached[I] = flood_update(I, scratch, near)
        reached == scratch && break # converged
    end
    @. scratch = !near && !reached # far-inside
end
@inline function flood_update(I::CartesianIndex{d}, r, blocked) where d
    blocked[I] && return false
    f = r[I]
    for i in 1:d
        f = f || r[I+δ(i,I)] || r[I-δ(i,I)]
    end; return f
end
