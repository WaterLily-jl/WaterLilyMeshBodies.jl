using WaterLily, WaterLilyMeshBodies, StaticArrays, WriteVTK

function make_sim(L=32;U=1,Re=500,T=Float32,mem=Array)
    # make a cube (exact) body
    # body = MeshBody(joinpath(@__DIR__,"../test/meshes/box.stl"); scale=T(L/2), boundary=true)
    body = MeshBody(joinpath(@__DIR__,"../test/meshes/sphere.stl"); scale=T(L), boundary=true)
    # move the mesh it to the center of the domain
    body = update!(body, [t .+ SA{T}[3L/2,3L/2,3L/2] for t in body.mesh], 0)
    # make the sim
    Simulation((6L,3L,3L),(U,0,0),L; body, ν=U*L/Re, T, mem)
end

# make a sim and some surface forces
sim = make_sim()
sf = SurfaceForces(sim.body)

# flow output functions for VTK
vtk_velocity(a::AbstractSimulation) = a.flow.u |> Array;
vtk_pressure(a::AbstractSimulation) = a.flow.p |> Array;
vtk_body(a::AbstractSimulation) = (measure_sdf!(a.flow.σ, a.body, WaterLily.time(a.flow)); a.flow.σ |> Array;)
custom_attrib = Dict("u"=>vtk_velocity, "p"=>vtk_pressure, "d"=>vtk_body)
wr = vtkWriter("surface_force"; attrib=custom_attrib)

# mesh output functions for VTK
mesh_velocity(a) = [SVector(sum(tri,dims=2)/3) for tri in Array(a.velocity)]
surface_pressure(a) = [SVector{3}(r) for r in eachrow(sf.pressure)] |> Array
surface_viscous(a) = [SVector{3}(r) for r in eachrow(sf.viscous)] |> Array
mesh_attrib = Dict("p"=>surface_pressure, "v"=>surface_viscous, "u"=>mesh_velocity)
wr_msh = vtkWriter("surface_force_mesh"; attrib=mesh_attrib)

# run a convective time
data = []
@time for tᵢ in 0:0.05:6
    # update until the next time step
    @show tᵢ
    sim_step!(sim, tᵢ)
    # compute forces
    f_p  = 2WaterLily.pressure_force(sim)/sim.L^2    # classic WaterLily
    f_s  = 2WaterLily.viscous_force(sim)/sim.L^2
    sf_s = 2WaterLily.viscous_force(sf, sim)/sim.L^2 # using the SurfaceForces struct
    sf_p = 2WaterLily.pressure_force(sf, sim)/sim.L^2
    push!(data, (tᵢ, f_p[1], f_s[1], sf_p[1], sf_s[1]))
    # save
    save!(wr, sim); save!(wr_msh, sim.body, sim_time(sim));
end
close(wr); close(wr_msh)

using Plots
plot(getindex.(data,1), [getindex.(data,2), getindex.(data,3), getindex.(data,4), getindex.(data,5)],
     label=["Pressure force (WaterLily)" "Viscous force (WaterLily)" "Pressure force (SurfaceForces)" "Viscous force (SurfaceForces)"],
     xlabel="Time", ylabel="Force", lw=2)