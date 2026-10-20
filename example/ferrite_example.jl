using WaterLily, FerriteShells, WaterLilyMeshBodies, WriteVTK
import Ferrite # the MeshBody conversion only needs Ferrite, not FerriteShells

# a flat plate in the x-y plane
function make_plate(dims; L=10.0, W=L)
    corners = [Ferrite.Vec{2}((0.0, 0.0)), Ferrite.Vec{2}((L, 0.0)),
               Ferrite.Vec{2}((L, W)),     Ferrite.Vec{2}((0.0, W))]
    grid = shell_grid(Ferrite.generate_grid(Ferrite.QuadraticQuadrilateral, dims[1:2], corners))
    return grid
end

# unit cube with origin at (0,0,0) and side length L
function make_cube(dims; L=1.0)
    left,right = zero(Ferrite.Vec{3}), L*ones(Ferrite.Vec{3})
    return Ferrite.generate_grid(Ferrite.Tetrahedron, dims, left, right)
end

function make_sim(L=32;U=1,Re=500,T=Float32,mem=Array,make_solid=make_cube)
    # a volume grid: only the wet (outer) surface is meshed
    cube = make_solid((16,16,16); L=L)
    # move to center of domain
    body = MeshBody(cube; map=(x,t)->x.-L, boundary=true, scale=1.f0, mem=mem)
    # make the sim and return it
    Simulation((4L,3L,3L), (1,0,0), L; body, ν=U*L/Re, T, mem)
end

using GLMakie,Meshing#,CUDA
# CUDA.allowscalar(false)
sim = make_sim(32;U=1,Re=1500,T=Float32,mem=Array,make_solid=make_cube)
viz!(sim,duration=4,step=0.01,remeasure=false,
     fig_size=(1200,800),video="Ferrite_WaterLily.mp4",
     colormap=:linear_blue_95_50_c20_n256,colorrange=(0.15,0.5),algorithm=:mip,body_color=:white,
     body2mesh=true,hidedecorations=true,azimuth=-3π/4,elevation=π/6)