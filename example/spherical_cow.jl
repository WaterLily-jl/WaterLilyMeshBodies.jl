using WaterLily, StaticArrays, BiotSavartBCs, WaterLilyMeshBodies

function spherical_cow(;L=64, Re=1e6, U=1, T=Float32, mem=Array)
    # First load the mesh and compute its bounding box to determine scaling and centering
    mesh_path = joinpath(@__DIR__, "spherical_cow.stl")
    probe = MeshBody(mesh_path)
    lo, up = probe.bvh.nodes[1].lo, probe.bvh.nodes[1].up
    scale = T(L/maximum(up .- lo))
    mesh_center = scale * SVector{3}(lo .+ up)/2

    # create the RigidMap, MeshBody and Simulation
    center,θ,ω = 3L÷4 .- mesh_center, SA{T}[0,0,0], SA{T}[-0.0025,0,0.0025]
    map = RigidMap(center, θ; xₚ = mesh_center, ω)
    body = MeshBody(mesh_path; map, scale, mem)
    return BiotSimulation((3L÷2, 2L, 3L÷2), (0,U,0), L; body, T, ν=U*L/Re, mem)
end
function WaterLily.measure!(sim::WaterLily.AbstractSimulation, t = sum(sim.flow.Δt))
    # Measure state
    dt = sim.flow.Δt[end]; T = typeof(dt)
    θ,ω = sim.body.map.θ,sim.body.map.ω
    M = WaterLily.pressure_moment(sim.body.map.x₀ + sim.body.map.xₚ, sim)

    # Compute angular acceleration, integrate, and update the map
    α =  -SVector{3, T}(M)/ T(π/60*sim.L^5)
    ω += dt * α; θ += dt * ω
    sim.body = setmap(sim.body; θ, ω)

    # measure the updated body and update the poisson coefficients
    WaterLily.measure!(sim.flow,sim.body;t,ϵ=sim.ϵ)
    WaterLily.update!(sim.pois)
end

using GLMakie, Meshing, CUDA
Makie.inline!(false)
CUDA.allowscalar(false)

sim = spherical_cow(L=32, mem=CuArray);
viz!(sim, body2mesh=true, remeasure=true,
    azimuth=-0.5, fig_size=(1200, 800),
    duration=4, step=0.02,# video="spherical_cow.mp4",
    colorrange=(0.2, 0.8), colormap=:dense, body_color=:white)