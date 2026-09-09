# Sphere drag benchmark for the `SurfaceForces` estimators.
#
# Flow past a sphere is steady and axisymmetric up to Re ≈ 210 (Johnson & Patel, JFM 378, 1999)
# with Cd(100) ≈ 1.09, and the friction share of the drag falls monotonically from exactly 2/3
# in the Stokes limit. Those give us a total to hit and a ceiling the friction fraction must sit
# under. We report the drag split by both routes:
#
#   WL    WaterLily's grid integral, -2ν S·nds summed over the BDIM band
#   SF    the mesh integral, `SurfaceForces`, using whatever stencil `src/forces.jl` holds
#   SFold the superseded stencil, anchored on the surface at the body velocity vₑ
#
# The open question this run is meant to settle: SF and WL are both first-order in h and were
# still climbing at D=32, extrapolating to different limits (≈0.42 vs ≈0.59 for Cd,v at Re=100).
# D=64 puts the boundary layer at δ_BL ≈ D/√Re ≈ 6.4 cells, comfortably thicker than the 2-cell
# stencil, so it should show which extrapolation was real.
#
# Usage
#   julia --project=. example/sphere_validation.jl --gpu --D=32,64 --Re=100 --tend=40
#   julia --project=. example/sphere_validation.jl --D=16 --tend=5          # quick CPU smoke test
#
# --D     comma-separated resolutions, cells per sphere DIAMETER   (default 32,64)
# --Re    comma-separated Reynolds numbers based on the diameter   (default 100)
# --dom   domain size in diameters, "Lx,Ly,Lz"                     (default 8,4,4)
# --tend  end time in convective units D/U                         (default 40)
# --gpu   run on CUDA (needs CUDA.jl in the project)
#
# NOTE: requires the shifted-stencil `get_v` in src/forces.jl (samples δ, 1.5δ, 2δ; no vₑ).
# The script prints the loaded package path so you can check which copy you are running.

using WaterLily, WaterLilyMeshBodies, StaticArrays, Printf
using WaterLily: interp
using WaterLilyMeshBodies: center, dS, hat, proj, area, get_velocity

# ---------------------------------------------------------------- arguments
const opt = Dict{String,String}()
for a in ARGS
    m = match(r"^--([^=]+)(?:=(.*))?$", a)
    m === nothing || (opt[m[1]] = something(m[2], "true"))
end
list(k, d) = haskey(opt, k) ? parse.(Float64, split(opt[k], ',')) : d
num(k, d)  = haskey(opt, k) ? parse(Float64, opt[k]) : d

const Ds   = list("D",   [32.0, 64.0])
const Res  = list("Re",  [100.0])
const dom  = Tuple(Int.(list("dom", [8.0, 4.0, 4.0])))
const tend = num("tend", 40.0)
const T    = Float32

mem = Array
if haskey(opt, "gpu")
    using CUDA
    CUDA.allowscalar(false)
    mem = CuArray
end

# ---------------------------------------------------------------- geometry
const GB  = WaterLilyMeshBodies.GeometryBasics
const STL = normpath(joinpath(@__DIR__, "..", "test", "meshes", "sphere.stl"))

# `sphere.stl` is the unit sphere: scaled by D it has DIAMETER D. Translate the points before
# the device transfer -- `update!` cannot mix a host mesh into a device body, and `map` is not
# an option here because the surface forces sample the flow at the mesh coordinates and ignore it.
function sphere_body(D, centre; mem)
    msh = WaterLilyMeshBodies.load(STL)
    pts = [GB.Point{3,T}(T.(D .* Tuple(p) .+ Tuple(centre))) for p in GB.coordinates(msh)]
    MeshBody(GB.Mesh(pts, GB.faces(msh)); boundary=true, mem)
end

# ---------------------------------------------------------------- the superseded stencil
# kept as a reference column; kernel-based so it runs on the GPU like the shipped one
@inline function get_v_old(tri::SMatrix{3,3,T},vel,u::AbstractArray{T,4},ν,δ) where T
    c = center(tri); ds = dS(tri); n = hat(ds)
    vₑ = get_velocity(c, tri, vel)
    v₁ = interp(c + δ*n, u); v₂ = interp(c + 2δ*n, u)
    return -ν*area(ds)*proj((4v₁ - v₂ - 3vₑ)/2δ, n)
end
function force_old(sim, F, δ)
    @WaterLily.loop F[I,:] .= get_v_old(sim.body.mesh[I], sim.body.velocity[I],
                                        sim.flow.u, sim.flow.ν, δ) over I in CartesianIndices(1:size(F,1))
    sum(Float64, F, dims=1)[:] |> Array
end

# ---------------------------------------------------------------- one case
function run(D, Re)
    U = one(T); δ = one(T)
    N = dom .* Int(D)
    # upstream margin (Lx-Ly)/2 diameters, sphere on the axis
    centre = SA{T}[(dom[1]-dom[2])/2*D, dom[2]/2*D, dom[3]/2*D]
    body = sphere_body(T(D), centre; mem)
    sim  = Simulation(N, (U,zero(U),zero(U)), T(D); body, ν=T(U*D/Re), T, mem) # homogeneous tuple: a mixed
    # Tuple{Float32,Int64,Int64} cannot be indexed inside WaterLily's applyV! kernel on the GPU
    sf   = SurfaceForces(sim.body)
    Fold = zeros(T, length(sim.body.mesh), 3) |> mem

    cd(F) = 8F/(π*U^2*D^2)          # Cd = F / (½ρU²·πD²/4)
    @printf("\n=== D=%d  Re=%g  domain=%s (%d cells)  δ_BL≈%.1f cells  t→%g ===\n",
            Int(D), Re, string(N), prod(N), D/√Re, tend)
    @printf("%6s | %7s %7s %7s %5s | %7s %7s %7s %5s | %7s\n",
            "t", "WL p", "WL v", "WL Cd", "f%", "SF p", "SF v", "SF Cd", "f%", "SFold v")
    out = nothing
    for tᵢ in 0:2.0:tend
        sim_step!(sim, tᵢ)
        pw = -cd(WaterLily.pressure_force(sim)[1]);      vw = -cd(WaterLily.viscous_force(sim)[1])
        ps = -cd(WaterLily.pressure_force(sf,sim)[1]);   vs = -cd(WaterLily.viscous_force(sf,sim)[1])
        vo = -cd(force_old(sim, Fold, δ)[1])
        out = (D, Re, pw, vw, ps, vs, vo)
        tᵢ % 4 < 2 && @printf("%6.1f | %7.4f %7.4f %7.4f %5.1f | %7.4f %7.4f %7.4f %5.1f | %7.4f\n",
                tᵢ, pw, vw, pw+vw, 100vw/(pw+vw), ps, vs, ps+vs, 100vs/(ps+vs), vo)
        flush(stdout)
    end
    out
end

# ---------------------------------------------------------------- go
@printf("WaterLilyMeshBodies: %s\nbackend: %s\n", pathof(WaterLilyMeshBodies), mem)
results = [run(D, Re) for Re in Res, D in Ds]

println("\n\n########## summary (converged values) ##########")
@printf("%5s %5s | %7s %7s %7s %5s | %7s %7s %7s %5s | %7s\n",
        "Re", "D", "WL p", "WL v", "WL Cd", "f%", "SF p", "SF v", "SF Cd", "f%", "SFold v")
for (D,Re,pw,vw,ps,vs,vo) in results
    @printf("%5g %5d | %7.4f %7.4f %7.4f %5.1f | %7.4f %7.4f %7.4f %5.1f | %7.4f\n",
            Re, Int(D), pw, vw, pw+vw, 100vw/(pw+vw), ps, vs, ps+vs, 100vs/(ps+vs), vo)
end
println("""
reference: Cd(Re=100) ≈ 1.09 (Johnson & Patel 1999); Schiller-Naumann Cd = 24/Re(1+0.15Re^0.687)
           friction share is exactly 2/3 as Re→0 and falls monotonically, so it is a ceiling
what to look for: does SF v stop climbing with D, and do SF and WL approach each other or not""")
