using WaterLilyMeshBodies
using Test, GPUArrays, StaticArrays, WaterLily, LinearAlgebra
import ImplicitBVH
import ImplicitBVH: BBox, BSphere

# Test utility: brute-force closest point search (moved from src/bvh.jl)
@fastmath @inline d²_fast(x::SVector,tri::SMatrix) = sum(abs2,x-WaterLilyMeshBodies.locate(x,tri))

# Conditionally use CUDA if available
arrays = [Array]  # Default to CPU-only
try
    using CUDA
    if CUDA.functional()
        push!(arrays, CuArray)
        @info "CUDA detected and functional - running tests on CPU and GPU"
    else
        @info "CUDA detected but not functional - running tests on CPU only"
    end
catch
    @info "CUDA not available - running tests on CPU only"
end

T = Float32
mem = Array
tri1 = SA{T}[0 1 0; 0 0 1; 0 0 0]
R = SA{T}[cos(π/4) -sin(π/4) 0; sin(π/4) cos(π/4) 0; 0 0 1]

@testset "Geometry Functions" begin
    normal = WaterLilyMeshBodies.normal
    @test all(normal(tri1) .≈ [0,0,1])
    @test all(normal(R*tri1) .≈ R*normal(tri1))

    hat = WaterLilyMeshBodies.hat
    vec = SA{T}[3,4,0]
    @test all(hat(vec) .≈ vec./5)
    @test all(hat(zero(vec)) .≈ zero(vec)) # edge case

    @test d²_fast(SA{T}[0.1,0.1,0.1], tri1) ≈ 0.1^2
    @test d²_fast(SA{T}[0.5,0.5,0.0], tri1) ≈ 0^2
    @test d²_fast(R*SA{T}[0.1,0.1,0.1], R*tri1) ≈ 0.1^2 # invariant under rotation

    center = WaterLilyMeshBodies.center
    @test all(center(tri1) .≈ SA{T}[1/3,1/3,0])
    @test all(abs.(center(R*tri1) .- R*SA{T}[1/3,1/3,0]) .< eps(Float32))

    locate = WaterLilyMeshBodies.locate
    x1 = SA{T}[0,0,0]
    @test all(locate(x1, tri1) .≈ x1)
    x2 = SA{T}[0.1,0.1,0.0]
    @test all(locate(x2, tri1) .≈ x2)
    @test all(locate(x2.+SA{T}[0,0,10.0], tri1) .≈ x2)
    x3 = SA{T}[-1.0,0.5,0.0]
    @test all(locate(x3, tri1) .≈ SA{T}[0.0,0.5,0.0])
    x4 = SA{T}[0.5,0.5,0.0]
    @test all(locate(x4, tri1) .≈ x4)
    @test all(locate(SA{T}[.5,.5,10], tri1) .≈ x4)
    @test all(locate(SA{T}[1,1,1], tri1) .≈ [0.5,0.5,0.0])
end

@testset "Interpolation" begin
    shape_value = WaterLilyMeshBodies.shape_value
    tri = SA{T}[0 1 0; 0 0 1; 0 0 0]
    p = SA{T}[0.1,0.1,0.0]
    # value at nodes
    @test all(shape_value(SA{T}[0,0,0], tri) .≈ [1,0,0])
    @test all(shape_value(SA{T}[0,1,0], tri) .≈ [0,0,1])
    @test all(shape_value(SA{T}[1,0,0], tri) .≈ [0,1,0])
    # value at mid edges
    @test all(shape_value(SA{T}[0,.5,0], tri) .≈ [.5,0,.5])
    @test all(shape_value(SA{T}[.5,0,0], tri) .≈ [.5,.5,0])
    @test all(shape_value(SA{T}[.5,.5,0], tri) .≈ [0,.5,.5])
    # value inside
    x = rand() # in the plane of the triangle
    @test sum(shape_value(SA{T}[x,1-x,0], tri)) .≈ 1 # partition of unity

    get_velocity = WaterLilyMeshBodies.get_velocity
    vel = SA{T}[1 1 1; 0 0 0; 0 0 0]
    @test all(get_velocity(p, tri, vel) .≈ [1,0,0])
    @test all(get_velocity(p, tri, R*vel) .≈ R*[1,0,0])
end

@testset "BVH Traversal" begin
    closest = WaterLilyMeshBodies.closest
    x1 = SA{T}[0,0,0]

    for mem in arrays
        mesh = mem([tri1, R*tri1 .+ 1])
        bounding_boxes = BBox{T}.(mesh)
        bvh = ImplicitBVH.BVH(bounding_boxes, BBox{T})

        # trivial locate
        @test GPUArrays.@allowscalar (c=closest(x1,bvh,mesh); c.d²≈0 && c.index==1)
        @test GPUArrays.@allowscalar (c=closest(SA{T}[1,1,1],bvh,mesh); c.d²≈0 && c.index==2)
        # not so trivial
        @test GPUArrays.@allowscalar (c=closest(SA{T}[0.1,0.1,0.5],bvh,mesh); c.d²≈0.5^2 && c.index==1)
    end
end

@testset "Sharp Edge Sign Consistency" begin
    # exterior
    mesh = [SA{T}[0 0 1; 0 0 0; 1 0 0],SA{T}[0 0 1; 1 0 0; 0 1 0]]
    bvh = ImplicitBVH.BVH(BBox{T}.(mesh), BBox{T})
    body = MeshBody(mesh, zero(mesh), bvh; boundary=true)
    @test sdf(body,SA{T}[1,0.1,1],0f0)>0
    # interior
    mesh = [SA{T}[0 0 1; 0 0 0; 0 1 0],SA{T}[0 0 1; 0 1 0; 1 0 0]]
    bvh = ImplicitBVH.BVH(BBox{T}.(mesh), BBox{T})
    body = MeshBody(mesh, zero(mesh), bvh; boundary=true)
    @test sdf(body,SA{T}[1,0.1,1],0f0)<0
end

@testset "measure & sdf" begin
    measure = WaterLily.measure
    x1 = SA{T}[0,0,0]
    x2 = SA{T}[0.1,0.1,0.0]

    for mem in arrays
        mesh = mem([tri1, R*tri1 .+ 1])
        bounding_boxes = BBox{T}.(mesh)
        bvh = ImplicitBVH.BVH(bounding_boxes, BBox{T})
        body = MeshBody(mesh, zero(mesh), bvh, half_thk=0f0)

        @test GPUArrays.@allowscalar all(body.bvh.nodes[1].lo .≈ [0,0,0]) # lowest point of tri1
        @test GPUArrays.@allowscalar all(measure(body, x1, 0) .≈ (0,[0,0,1],[0,0,0]))
        @test GPUArrays.@allowscalar all(isapprox.(measure(body, x2, 0),(0,[0,0,1],[0,0,0]),atol=1e-6))
        @test GPUArrays.@allowscalar all(measure(body, SA{T}[.5,.5,100], 0, fastd²=16f0) .≈ (4,[0,0,0],[0,0,0]))
        xr = SVector{3,T}(rand(3))
        @test GPUArrays.@allowscalar measure(body, xr, 0)[1] ≈ sdf(body, xr, 0, fastd²=Inf32)
    end
end

@testset "Flood Classifier" begin
    sdf = fill(T(5), 16, 16)
    cutoff = T(4)
    # Closed near-band ring encloses a 6x6 interior region (indices 6:11, 6:11)
    sdf[5, 5:12] .= 0
    sdf[12, 5:12] .= 0
    sdf[5:12, 5] .= 0
    sdf[5:12, 12] .= 0
    near = similar(sdf, Bool)
    reached = similar(sdf, Bool)
    farinside = similar(sdf, Bool)
    WaterLilyMeshBodies.flood_fill!(near, reached, farinside, sdf; cutoff)
    @test count(@view(farinside[6:11, 6:11])) == 36
    @test all(@view(farinside[5, 5:12]) .== false)
    @test all(@view(farinside[12, 5:12]) .== false)
end

@testset "measure_sdf!" begin
    L = 32
    R = 0.707f0L
    for mem in arrays
        mesh_body = MeshBody(joinpath(@__DIR__, "meshes", "sphere.stl");
            scale=1.414f0L, map=(x,t)->x .- L, boundary=true, mem)
        σm = mem{T}(undef, 2L+2,2L+2,2L+2)
        measure_sdf!(σm, mesh_body, 0f0, fastd²=16f0)

        auto_body = AutoBody((x,t) -> √sum(abs2, x .- L) - R)
        σa = mem{T}(undef, 2L+2,2L+2,2L+2)
        measure_sdf!(σa, auto_body, 0f0, fastd²=16f0)

        # Any discrepancy should be due to triangle discretization error
        v,I = findmax(abs.(clamp.(σm,-4,4) - clamp.(σa,-4,4)))
        ξ = mesh_body.map(SVector{3,T}(WaterLily.loc(0, I, T)), 0f0)
        GPUArrays.@allowscalar (;p) = WaterLilyMeshBodies.closest(ξ, mesh_body.bvh, mesh_body.mesh)
        @test v ≈ R-√(p'p) atol=√eps(T)

        mismatches = findall(signbit.(σm) .!= signbit.(σa))
        @test GPUArrays.@allowscalar all(0>σa[I]>-v && 0<σm[I]<v for I in mismatches) # all sign mismatches must be on the boundary
    end
end

@testset "Updates" begin
    x1 = SA{T}[0,0,0]

    for mem in arrays
        mesh = mem([tri1, R*tri1 .+ 1])
        bounding_boxes = BBox{T}.(mesh)
        bvh = ImplicitBVH.BVH(bounding_boxes, BBox{T})
        body = MeshBody(mesh, zero(mesh), bvh, half_thk=0f0)

        # update! by moving the mesh by +1 in all directions
        new_mesh = mem([tri1 .+ 1, R*tri1 .+ 2])
        body = update!(body, new_mesh, 1.0)
        @test GPUArrays.@allowscalar all(body.velocity[1] .≈ 1) && all(body.velocity[2] .≈ 1)
        @test GPUArrays.@allowscalar all(measure(body, x1.+1, 0) .≈ (0,[0,0,1],[1,1,1]))
        # check that bvh has also moved
        @test GPUArrays.@allowscalar all(body.bvh.nodes[1].lo .≈ [1,1,1])

        # try inside SetBody
        body += AutoBody((x,t)->42.f0) # the answer!
        @test GPUArrays.@allowscalar all(measure(body, x1.+1, 0) .≈ (0,[0,0,1],[1,1,1]))
    end
end

@testset "Simulation" begin
    L = 8
    for mem in arrays
        body = MeshBody(joinpath(@__DIR__, "meshes", "sphere.stl");
            scale = T(L), map = (x,t) -> x - SA[L,0,0], mem)
        sim = Simulation((2L, L, L), (1,0,0), L; body, T, ν=1e-3, mem)
        sim_step!(sim, 0.1, remeasure=false)
        @test maximum(sim.pois.n) < 10
        @test 1 > sim.flow.Δt[end] > 0
    end
end

@testset "RigidMap MeshBody" begin
    if @isdefined(CuArray) && (CuArray in arrays)
        mesh_file = joinpath(@__DIR__, "meshes", "sphere.stl")
        center = SA{T}[0, 0, 0]
        theta = SA{T}[0, 0, 0]
        map = RigidMap(center, theta; xₚ=center)
        body = MeshBody(mesh_file; scale=T(8), map, mem=CuArray)
        converted = CUDA.cudaconvert(body)
        @test converted isa WaterLilyMeshBodies.MeshBody
        @test converted.map === map
    else
        @test_skip "CUDA backend unavailable; skipping GPU-only RigidMap adaptation regression"
    end
end