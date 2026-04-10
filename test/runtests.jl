using WaterLilyMeshBodies
using Test, GPUArrays, StaticArrays, WaterLily, LinearAlgebra, GeometryBasics
import ImplicitBVH
import ImplicitBVH: BBox, BSphere

# Test utility: brute-force closest point search (moved from src/bvh.jl)
@fastmath @inline d²_fast(x::SVector,tri::SMatrix) = sum(abs2,x-WaterLilyMeshBodies.locate(x,tri))

# Conditionally use CUDA if available
arrays = [Array]  # Default to CPU-only
try
    using CUDA
    if CUDA.functional() && WaterLily.backend != "SIMD"
        push!(arrays, CuArray)
        @info "Running tests on CPU and GPU"
    end
catch
    @info "Running tests on CPU only"
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
    # Closed near-band ring encloses a 6x6 interior region (indices 6:11, 6:11)
    d = fill(T(5), 16, 16)
    d[5, 5:12] .= 0
    d[12, 5:12] .= 0
    d[5:12, 5] .= 0
    d[5:12, 12] .= 0
    near = similar(d, Bool)
    reached = similar(d, Bool); fill!(reached, true); reached[inside(d)] .= false
    farinside = similar(d, Bool)
    WaterLilyMeshBodies.flood_fill!(near, reached, farinside, d)
    @test count(@view(farinside[6:11, 6:11])) == 36
    @test all(@view(farinside[5, 5:12]) .== false)
    @test all(@view(farinside[12, 5:12]) .== false)
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

@testset "measure_sdf!" begin
    L = 16; R = 0.707f0L; size = (2L, 2L, 2L)
    fastd²=9f0; cutoff = sqrt(fastd²)
    for mem in arrays
        # Compare MeshBody SDF to AutoBody SDF for a sphere
        mesh_body = MeshBody(joinpath(@__DIR__, "meshes", "sphere.stl");
            scale=2R, map=(x,t)->x .- L, boundary=true, mem)
        σm = zeros(T,size .+ 2) |>  mem
        measure_sdf!(σm, mesh_body, 0f0; fastd²)
        @test mesh_body.cache === nothing 

        auto_body = AutoBody((x,t) -> √sum(abs2, x .- L) - R)
        σa = zeros(T,size .+ 2) |>  mem
        measure_sdf!(σa, auto_body, 0f0; fastd²)

        # Any discrepancy should be due to triangle discretization error
        v,I = findmax(abs.(σm - clamp.(σa,-cutoff,cutoff)))
        ξ = mesh_body.map(SVector{3,T}(WaterLily.loc(0, I, T)), 0f0)
        GPUArrays.@allowscalar (;p) = WaterLilyMeshBodies.closest(ξ, mesh_body.bvh, mesh_body.mesh)
        @test v ≈ R-√(p'p) atol=√eps(T)

        # all sign mismatches must be on the boundary
        mismatches = findall(signbit.(σm) .!= signbit.(σa))
        @test GPUArrays.@allowscalar all(0>σa[I]>-v && 0<σm[I]<v for I in mismatches) 

        # test caching 
        cache_body = MeshBody(joinpath(@__DIR__, "meshes", "sphere.stl");
            scale=2R, map=(x,t)->x .- L, boundary=true, mem, size)
        σc = zeros(T,size .+ 2) |>  mem
        @test !isnothing(cache_body.cache)

        measure_sdf!(σc, cache_body, 0f0; fastd²)
        @test σc ≈ σm
        num_near,num_reached,num_farinside = count.(cache_body.cache)
        @test num_farinside ≈ 4π/3*R^3-4π*R^2 rtol = 0.05 # should be close to the number of points in the interior

        # shift the mesh by 1/2 cell and check that cache persists
        cache_body = update!(cache_body, [tri .+ 0.5 for tri in cache_body.mesh], 1f0)
        abs_vel(vel) = maximum(√sum(abs2,vertex) for vertex in eachcol(vel))
        @test maximum(abs_vel.(cache_body.velocity)) < 1 # can't shift by more than 1 cell in one time step
        @test all((num_near,num_reached,num_farinside) .== count.(cache_body.cache))

        # warm-start should give same farinside count and same result after an integer shift
        measure_sdf!(σc, cache_body, 1f0; fastd²)
        cache_body = update!(cache_body, [tri .+ 0.5 for tri in cache_body.mesh], 1f0)
        measure_sdf!(σc, cache_body, 1f0; fastd²)
        @test num_farinside == count(cache_body.cache[3])
        @test σc[3:2L-1,3:2L-1,3:2L-1] ≈ σm[2:2L-2,2:2L-2,2:2L-2]
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
        # test with SetBody
        body += AutoBody((x,t)->42.f0)
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

@testset "Quad mesh" begin
    L = 8
    for mem in arrays
        rect = Rect((0.f0, 0.f0, 0.f0), (1.f0, 1.f0, 1.f0))
        points = decompose(Point{3, Float32}, rect)
        faces = decompose(QuadFace{Int}, rect)
        mesh = GeometryBasics.Mesh(points, faces)
        body = MeshBody(mesh; scale = T(L/4.f0), map = (x,t) -> x - SA_F32[L,L÷3,L÷3], mem)
        sim = Simulation((2L, L, L), (1,0,0), L; body, T, ν=1e-3, mem)
        sim_step!(sim, 0.1, remeasure=false)
        @test maximum(sim.pois.n) < 10
        @test 1 > sim.flow.Δt[end] > 0
    end
end