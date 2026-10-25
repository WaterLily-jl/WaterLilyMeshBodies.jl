# Surface meshes of WaterLily bodies and walls, and the flow on them

import Meshing
"""
    MeshBody(body::AbstractBody, N::NTuple{3}; t=0, mem=Array, boundary=true, kwargs...)

Marching-cubes mesh of the zero level set of `body` on the `N` grid, with the body velocity at the vertices.
"""
function MeshBody(body::AbstractBody, N::NTuple{3,Int}; t=0f0, mem=Array, boundary=true, kwargs...)
    σ = mem(zeros(Float32, N.+2)); measure_sdf!(σ, body, t)
    pts, fcs = Meshing.isosurface(Array(σ)[2:end-1,2:end-1,2:end-1], ntuple(i->(0.5,N[i]-0.5),3)...)
    b = MeshBody(GeometryBasics.Mesh(Point3f.(pts), GLTriangleFace.(fcs)); mem, boundary, kwargs...)
    copyto!(b.velocity, [hcat((measure(body,tri[:,i],t)[3] for i in 1:3)...) for tri in Array(b.mesh)]); b
end
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
"""
    vertices(body::MeshBody)

Positions and normals of the triangle vertices as `3K×3` matrices, in mesh coordinates and in the order of `save!`.
"""
vertices(a::MeshBody{T}) where T = rows(a.mesh,T), rows(repeat(normal.(a.mesh),inner=3),T)
rows(v,T) = permutedims(reshape(reinterpret(T,v),3,:))
"""
    surface_sample(sim, body::MeshBody; walls=false, δ=1)

Pressure `p`, traction `τ` (the wall's on the fluid, as `total_force`), wall shear `ws` (the fluid's on the wall) and, for
bodies, slip velocity `slip` at the `3K` vertices of `body`. Walls are sampled on the surface, where no-slip is exact, and
smoothed bodies `δ` off it, outside the BDIM kernel, with the stencils of `SurfaceForces` (constant per triangle).
"""
function surface_sample(sim,b::MeshBody;walls=false,δ=1)
    x,n = vertices(b)
    if walls
        v = WaterLily.surface_values(sim.flow,x,n)
        return (p=v.p, τ=v.τ, ws=n.*sum(v.τ.*n,dims=2).-v.τ)
    end
    ps,wss,sl = similar(x,size(x,1)),similar(x),similar(x); bnd = Val(b.boundary)
    @loop body_sample!(ps,wss,sl,I,b.mesh,b.velocity,sim.flow.p,sim.flow.u,sim.flow.ν,δ,bnd) over I ∈ CartesianIndices(b.mesh)
    (p=ps, τ=ps.*n.-wss, ws=wss, slip=sl)
end
@inline function body_sample!(ps,wss,sl,I,mesh,vel,p,u,ν,δ,bnd)
    tri = mesh[I]; ds = dS(tri); A = area(ds)
    pk,wk,sk = sum(get_p(tri,p,δ,bnd).*ds)/A^2, -get_v(tri,vel[I],u,ν,δ,bnd)/A, interp(center(tri),u)-vel[I]*SA[1,1,1]/3
    for j in 3I[1]-2:3I[1]; ps[j] = pk; wss[j,:] .= wk; sl[j,:] .= sk; end
end
"""
    SurfaceStats(sim, body::MeshBody; walls=false)

Running time averages of the pressure and wall shear on `body` and of their squares, for their mean and RMS. Call
`update!(stats,sim)` every time step and pass `stats` to `surface_attrib`.
"""
struct SurfaceStats{B,A,M}
    body :: B
    walls :: Bool
    p :: A; p² :: A; ws :: M; ws² :: M
    t :: Vector{Float64}
end
SurfaceStats(sim,b::MeshBody;walls=false) = (x = vertices(b)[1]; z(a) = fill!(similar(a),0);
    SurfaceStats(b,walls,z(x[:,1]),z(x[:,1]),z(x),z(x),[Float64(WaterLily.time(sim.flow))]))
function update!(s::SurfaceStats,sim)
    t = WaterLily.time(sim.flow); ε = (t-s.t[end])/(t-s.t[1]+eps()); push!(s.t,t)
    v = surface_sample(sim,s.body;s.walls)
    for (m,a) in ((s.p,v.p),(s.p²,v.p.^2),(s.ws,v.ws),(s.ws²,v.ws.^2)); @. m = ε*a+(1-ε)*m; end
end
rms(m,m²) = sqrt.(max.(m².-m.^2,0))
"""
    surface_attrib(sim; walls=false, stats=nothing)

`vtkWriter` attributes for `save!(w,body::MeshBody,t)`: pressure, traction, wall shear, slip velocity (bodies only) and normal
of `surface_sample`, and the mean and RMS pressure and wall shear of `stats::SurfaceStats`. Use `walls=true` for the walls.
"""
function surface_attrib(sim; walls=false, stats=nothing)
    s(f) = b->Array(f(surface_sample(sim,b;walls)))
    d = Dict{String,Function}("Pressure"=>s(v->v.p), "Traction"=>s(v->v.τ), "WallShear"=>s(v->v.ws), "Normal"=>b->Array(vertices(b)[2]))
    walls || (d["SlipVelocity"] = s(v->v.slip))
    isnothing(stats) || merge!(d, Dict{String,Function}("MeanPressure"=>_->Array(stats.p), "RMSPressure"=>_->Array(rms(stats.p,stats.p²)),
        "MeanWallShear"=>_->Array(stats.ws), "RMSWallShear"=>_->Array(rms(stats.ws,stats.ws²))))
    d
end
