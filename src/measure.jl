# Signed distance field and measure functions

using StaticArrays
using ForwardDiff
using WaterLily

# signed distance function
WaterLily.sdf(body::Meshbody,x,t;kwargs...) = measure(body,x,t;kwargs...)[1]

# measure
function WaterLily.measure(body::Meshbody,x::SVector{D,T},t;fastd²=Inf) where {D,T}
    # map to correct location
    ξ = body.map(x,t)
    # before we try the bvh
    !inside(ξ,body.bvh.nodes[1]) && return (T(4),zero(x),zero(x))
    # locate the point on the mesh
    nearest = closest(ξ,body.bvh,body.mesh;init_d²= body.boundary ? floatmax(T) : T(16))
    nearest.index==0 && return (T(4),zero(x),zero(x)) # no triangles within init_d²
    # signed Euclidian distance
    d = copysign(√nearest.d²,nearest.n'*(ξ-nearest.p))
    !body.boundary && (d = abs(d)-body.half_thk) # if the mesh is not a boundary, we need to adjust the distance
    d^2>fastd² && return (d,zero(x),zero(x)) # skip n,V
    # velocity at the mesh point
    dξdx = ForwardDiff.jacobian(x->body.map(x,t), ξ)
    dξdt = -ForwardDiff.derivative(t->body.map(x,t), t)
    # mesh deformation velocity
    v = get_velocity(nearest.p, body.mesh[nearest.index], body.velocity[nearest.index])
    return (d,dξdx\nearest.n,dξdx\dξdt+v)
end
