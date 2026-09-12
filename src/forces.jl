"""
    SurfaceForces

Hold the pressure (`pressure`) and viscous (`viscous`) surface forces on the `MeshBody`.

Unlike `measure`, which maps the query point into the mesh frame, the forces interpolate the
flow at the mesh coordinates themselves and cannot apply `body.map`. The mesh must therefore
be placed inside the flow domain, either where it is generated or by translating it with
`update!(body,new_mesh)`; a body positioned by a non-identity `map` samples the wrong place
and gives a wrong force.
"""
struct SurfaceForces{T,A}
    pressure :: A
    viscous :: A
    function SurfaceForces(body::MeshBody)
        @warn "the surface forces sample the flow at the mesh coordinates and ignore `body.map`, \
               the mesh must be placed inside the flow domain" maxlog=1
        T = basetype(body.mesh)
        mem = typeof(body.mesh).name.wrapper
        A = zeros(T,length(body.mesh),3) |> mem
        new{T,typeof(A)}(A,copy(A))
    end
end
export SurfaceForces

function WaterLily.pressure_force(a::SurfaceForces,sim::AbstractSimulation;kwargs...)
    Tp = eltype(a.pressure); To = promote_type(Float64,Tp)
    surface_pressure!(a,sim;kwargs...); sum(To,a.pressure,dims=1)[:] |> Array
end
"""
    sample_offset(body::MeshBody)

Default distance, in cells, from the mesh at which the surface samplers read the flow.

`measure!` truncates `μ₀` to zero wherever `sdf < -1`, and a cell blocked on every face
drops out of the Poisson operator entirely: nothing constrains its pressure and it drifts
(measured on a moving shell, |p| reached 1e7 inside while the fluid held |p| ≈ 1.5). The
samplers must therefore keep clear of `sdf ≤ -1`, and not just at the sample point --
`interp` is trilinear over the cell cube containing it, so a stencil corner can sit up to
`|n|₁ ≤ √3` cells further along `-n`.

For a shell the mesh is the *mid*-surface and `sdf = dist - half_thk`, so an offset of
`half_thk + 2` puts the worst-case corner at `sdf ≈ 0.27`, where `μ₀ ≈ 0.75`. The flat
`δ=1` this used to default to reads `sdf = 1 - half_thk`, i.e. straight out of the
unconstrained interior for any `half_thk ≥ 2`.

For a closed body the mesh *is* the surface and the historical `δ=1` is kept, since the
resultant scales with the offset (`∮(a⋅x + δ a⋅n) n dA = (V + 2δL²)a`) and the validation
cases are calibrated against it. Note that it is marginal on the same argument: a corner
of the stencil can reach `sdf = 1-√3 ≈ -0.73`, where `μ₀ ≈ 0.01`. Pass `δ=2` explicitly
for a closed body whose interior pressure is suspect.
"""
sample_offset(body::MeshBody) = body.boundary ? one(body.half_thk) : body.half_thk + 2
export sample_offset

function surface_pressure!(a::SurfaceForces,sim::AbstractSimulation;δ=sample_offset(sim.body),boundary=Val{sim.body.boundary}())
    @WaterLily.loop a.pressure[I,:] .= get_p(sim.body.mesh[I],sim.flow.p,δ,boundary) over I in CartesianIndices(1:size(a.pressure,1))
end

function WaterLily.viscous_force(a::SurfaceForces,sim::AbstractSimulation;kwargs...)
    Tp = eltype(a.viscous); To = promote_type(Float64,Tp)
    surface_shear!(a,sim;kwargs...); sum(To,a.viscous,dims=1)[:] |> Array
end
# NB the δ=1 default here is NOT `sample_offset`: for a shell it samples at
# `sdf = 1-half_thk`, `0.5-half_thk` and `-half_thk`, i.e. inside the body, and the comment
# below about the samples lying outside the kernel support then does not hold. Only the
# pressure path is used by the coupled flag runs; fix this before turning the viscous force
# on for a shell.
function surface_shear!(a::SurfaceForces,sim::AbstractSimulation;δ=1,boundary=Val{sim.body.boundary}())
    @WaterLily.loop a.viscous[I,:] .= get_v(sim.body.mesh[I],sim.body.velocity[I],sim.flow.u,sim.flow.ν,δ,boundary) over I in CartesianIndices(1:size(a.viscous,1))
end

import WaterLily: interp
@inline function get_p(tri::SMatrix{3,3,T},p::AbstractArray{T,3},δ,::Val{true}) where T
    c=center(tri); ds=dS(tri); n=hat(ds)
    return ds.*interp(c + δ*n, p) # only outside
end
@inline function get_p(tri::SMatrix{3,3,T},p::AbstractArray{T,3},δ,::Val{false}) where T
    c=center(tri); ds=dS(tri); n=hat(ds)
    return ds.*(interp(c + δ*n, p) - interp(c - δ*n, p)) # both sides
end

@fastmath @inline proj(a,n) = a .- sum(a.*n)*n # tangent component
# Velocity gradient at the wall from a 2nd-order one-sided difference anchored one `δ` off the
# surface, sampling `δ`, `1.5δ` and `2δ`. The BDIM field within |d|≲1 of the mesh is masked and
# carries a spurious slip, so a stencil anchored on the surface itself -- whether it uses the
# body velocity or the field's own value there -- reads a wall gradient that converges to the
# wrong constant. These three samples all lie outside the kernel support, and span only `δ` so
# that they stay inside the near-wall region of a boundary layer a few cells thick. The body
# velocity is not needed: a uniform surface velocity cancels out of the difference, and the
# body's motion reaches these points through the flow field itself.
@fastmath @inline shear(v₁,v₂,v₃,δ) = (-3v₁ + 4v₂ - v₃)/δ # spacing δ/2, derivative at `δ`
@fastmath @inline area(ds) = √(ds'*ds)
@inline function get_v(tri::SMatrix{3,3,T},vel,u::AbstractArray{T,4},ν,δ,::Val{true})  where T
    c=center(tri); ds=dS(tri); n=hat(ds)
    v₁ = interp(c + δ*n, u)
    v₂ = interp(c + 1.5f0δ*n, u)
    v₃ = interp(c + 2δ*n, u)
    return -ν*area(ds)*proj(shear(v₁,v₂,v₃,δ),n) # only outside, projects once
end
@inline function get_v(tri::SMatrix{3,3,T},vel,u::AbstractArray{T,4},ν,δ,::Val{false})  where T
    c=center(tri); ds=dS(tri); n=hat(ds)
    τ = zero(SVector{3,T})
    for j ∈ (-1,1) # the outward direction of each side is j*n
        v₁ = interp(c + j*δ*n, u)
        v₂ = interp(c + j*1.5f0δ*n, u)
        v₃ = interp(c + j*2δ*n, u)
        τ = τ + shear(v₁,v₂,v₃,δ)
    end
    return -ν*area(ds)*proj(τ,n) # both sides, projects once
end
