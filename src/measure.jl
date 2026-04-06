# Signed distance field and measure functions

using StaticArrays
using ForwardDiff
using WaterLily
import WaterLily: @loop, δ, loc

# measure
function WaterLily.measure(body::Meshbody,x::SVector{D,T},t;fastd²=Inf) where {D,T}
    # map to correct location
    ξ = body.map(x,t)
    # locate the point on the mesh
    (;index,d²,n,p) = closest(ξ,body.bvh,body.mesh;init_d²= body.boundary ? floatmax(T) : T(16))
    index==0 && return (T(4),zero(x),zero(x)) # no triangles within init_d²
    # signed Euclidian distance
    d = copysign(√d²,n'*(ξ-p))
    !body.boundary && (d = abs(d)-body.half_thk) # if the mesh is not a boundary, we need to adjust the distance
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    # velocity at the mesh point
    dξdx = ForwardDiff.jacobian(x->body.map(x,t), ξ)
    dξdt = -ForwardDiff.derivative(t->body.map(x,t), t)
    # mesh deformation velocity
    v = get_velocity(p, body.mesh[index], body.velocity[index])
    return (d,dξdx\n,dξdx\dξdt+v)
end

function WaterLily.measure_sdf!(d::AbstractArray{T}, body::Meshbody, t=zero(T); fastd²=zero(T)) where T
    # Get the region of interest (ROI) from the BVH, and get the signed SDF within |d|≤cutoff
    cutoff = T(4)
    @inside d[I] = tightsdf(body, loc(0,I,T), t, cutoff)

    # If the mesh is not a boundary, adjust the distance and return 
    body.boundary || return @inside d[I] = abs(d[I]) - body.half_thk

    # Otherwise, flood-fill from the outside and make the distances negative inside the surface
    near,reached, farinside = similar(d, Bool), similar(d, Bool), similar(d, Bool)
    classify_from_sdf!(near, reached, farinside, d, cutoff)
    @inside d[I] = farinside[I] ? -abs(d[I]) : d[I]
end
@inline function tightsdf(body, x, t, cutoff)
    ξ = body.map(x, t)
    (;index,d²,n,p) = closest(ξ, body.bvh, body.mesh; init_d²=cutoff^2)
    index==0 ? cutoff : copysign(√d²,n'*(ξ-p))
end

# Flood-fill classification helpers for measure_sdf! (boundary=true fast path)
function classify_from_sdf!(near, reached, scratch, sdf, cutoff=4f0)
    near .= sdf .< cutoff
    r, s = reached, scratch
    fill!(r, true); r[inside(r)] .= false
    for _ in 1:max(0, min(size(r)...) ÷ 2)
        copyto!(s, r)
        @inside s[I] = flood_update!(I, r, near)
        r == s && break # converged
        r, s = s, r
    end
    r !== reached && copyto!(reached, r)
    @. scratch = !near && !reached
end
@inline function flood_update!(I::CartesianIndex{d}, r, blocked) where d
    blocked[I] && return false
    f = r[I]
    for i in 1:d
        f = f || r[I+δ(i,I)] || r[I-δ(i,I)]
    end; return f
end
