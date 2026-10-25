# Surface meshes of the walls (of bodies in ext/MeshingExt.jl), and the flow on them

"""
    MeshBody(w::Walls; kwargs...)

Mesh of the walls with two triangles per wall cell, vertices on the cell corners and normals into the fluid.
"""
function MeshBody(w::Walls{3}; kwargs...)
    tri = Point3f[]
    for g in w.faces, a in 0:w.N[mod1(abs(g)+1,3)]-1, b in 0:w.N[mod1(abs(g)+2,3)]-1
        P(s,r) = Point3f(circshift([g<0 ? 0 : w.N[abs(g)], a+s, b+r], abs(g)-1))
        append!(tri, g<0 ? (P(0,0),P(1,0),P(1,1),P(0,0),P(1,1),P(0,1)) : (P(0,0),P(1,1),P(1,0),P(0,0),P(0,1),P(1,1)))
    end
    MeshBody(GeometryBasics.Mesh(tri, [GLTriangleFace(i,i+1,i+2) for i in 1:3:length(tri)]); kwargs...)
end
# positions and normals of the triangle vertices as 3K×3 matrices, in mesh coordinates and in the order of `save!`
vertices(a::MeshBody{T}) where T = rows(a.mesh,T), rows(repeat(normal.(a.mesh),inner=3),T)
rows(v,T) = permutedims(reshape(reinterpret(T,v),3,:))
# fields of `surface_attrib` at the 3K vertices of `b`
function surface_sample(sim,b::MeshBody;δ=1)
    x,n = vertices(b); w = Walls(sim.flow)
    if all(tri->all(i->sdf(w,tri[:,i])==0,1:3),b.mesh) # on the walls
        v = WaterLily.surface_values(sim.flow,x,n)
        return (p=v.p, τ=v.τ, ws=n.*sum(v.τ.*n,dims=2).-v.τ, slip=zero(x), n)
    end
    ps,wss,sl = similar(x,size(x,1)),similar(x),similar(x); bnd = Val(b.boundary)
    @loop body_sample!(ps,wss,sl,I,b.mesh,b.velocity,sim.flow.p,sim.flow.u,sim.flow.ν,δ,bnd) over I ∈ CartesianIndices(b.mesh)
    (p=ps, τ=ps.*n.-wss, ws=wss, slip=sl, n)
end
@inline function body_sample!(ps,wss,sl,I,mesh,vel,p,u,ν,δ,bnd)
    tri = SMatrix{3,3,eltype(p)}(mesh[I]); ds = dS(tri); A = area(ds)
    pk,wk,sk = sum(get_p(tri,p,δ,bnd).*ds)/A^2, -get_v(tri,vel[I],u,ν,δ,bnd)/A, interp(center(tri),u)-center(vel[I])
    for j in 3I[1]-2:3I[1]; ps[j] = pk; wss[j,:] .= wk; sl[j,:] .= sk; end
end
"""
    surface_attrib(sim; δ=1)

`vtkWriter` attributes for `save!(w,body::MeshBody,t)`: pressure, traction (the wall's on the fluid, as `total_force`), wall shear
(the fluid's on the wall), slip velocity and normal. A mesh on the no-slip walls is sampled on the surface, where no-slip is exact,
and any other `δ` off it with the `SurfaceForces` stencils (constant per triangle), so both integrate to the force metrics. The
fields are linear in the flow, so after `WaterLily.copy!(sim.flow,meanflow)` they are the time averages on a static `body`.
"""
surface_attrib(sim; kw...) = Dict{String,Function}(k=>b->Array(surface_sample(sim,b;kw...)[f]) for (k,f) in
    ("Pressure"=>:p, "Traction"=>:τ, "WallShear"=>:ws, "SlipVelocity"=>:slip, "Normal"=>:n))
