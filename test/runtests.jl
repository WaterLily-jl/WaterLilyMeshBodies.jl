using WaterLilyMeshBodies
using Test, GPUArrays, StaticArrays, WaterLily
import ImplicitBVH
import ImplicitBVH: BBox, BSphere

# Test utility: brute-force closest point search (moved from src/bvh.jl)
@inline closest_brute(x::SVector,mesh) = findmin(tri->WaterLilyMeshBodies.d²_fast(x, tri),mesh)

# Conditionally use CUDA if available
arrays = [Array]  # Default to CPU-only
try
    using CUDA
    if CUDA.functional()
        arrays = [Array, CuArray]
        @info "CUDA detected and functional - running tests on CPU and GPU"
    else
        @info "CUDA detected but not functional - running tests on CPU only"
    end
catch
    @info "CUDA not available - running tests on CPU only"
end

@testset "WaterLilyMeshBodies.jl" begin
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

        d²_fast = WaterLilyMeshBodies.d²_fast
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

    @testset "Bounding Volumes" begin
        inside = WaterLilyMeshBodies.inside
        bbox = BBox{T}(SA{T}[-1.0,-1.0,-1.0], SA{T}[1.0,1.0,1.0])
        bsphere = BSphere{T}(SA{T}[0.0,0.0,0.0], T(1.0))
        @test inside(SA{T}[0.0,0.0,0.0], bbox)
        @test !inside(SA{T}[5.01,0.0,0.0], bbox)
        @test inside(SA{T}[0.0,0.0,0.0], bsphere)
        @test !inside(SA{T}[5.01,0.0,0.0], bsphere)
    end

    @testset "BVH Traversal" begin
        closest = WaterLilyMeshBodies.closest
        x1 = SA{T}[0,0,0]

        for mem in arrays
            mesh = mem([tri1, R*tri1 .+ 1])
            bounding_boxes = BBox{T}.(mesh)
            bvh = ImplicitBVH.BVH(bounding_boxes, BBox{T})

            # trivial locate
            @test GPUArrays.@allowscalar all(closest(x1, bvh, mesh) .≈ (0, 1))
            @test GPUArrays.@allowscalar all(closest(SA{T}[1,1,1], bvh, mesh) .≈ (0, 2))
            # not so trivial
            @test GPUArrays.@allowscalar all(closest(SA{T}[0.1,0.1,0.5], bvh, mesh) .≈ (0.5^2, 1))
        end
    end

    @testset "Measure & SDF" begin
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
            @test GPUArrays.@allowscalar all(measure(body, SA{T}[.5,.5,100], 0) .≈ (4,[0,0,0],[0,0,0]))
            xr = SVector{3,T}(rand(3))
            @test GPUArrays.@allowscalar all(measure(body, xr, 0)[1] .≈ sdf(body, xr, 0))
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

    @testset "Integration" begin
        closest = WaterLilyMeshBodies.closest
        locate = WaterLilyMeshBodies.locate
        test_points = [SA{T}[r*sin(φ)*cos(θ), r*sin(φ)*sin(θ), r*cos(φ)]
                       for r in T[0.2, 0.35, 0.65, 1.3], θ in T[0.7, 2.1, 4.3], φ in T[0.9, 1.8]]

        # for mem in arrays
        for mesh_file in ["sphere.stl","box.stl"]
            body = MeshBody(joinpath(@__DIR__, "meshes", mesh_file); scale=1.f0, mem, boundary=true)
            for x in test_points
                d_brute, u_brute = closest_brute(x, body.mesh)  # brute-force ground truth
                d_bvh, u_bvh = closest(x, body.bvh, body.mesh)  # BVH traversal
                @test d_bvh ≈ d_brute  # same squared distance
                @test u_bvh == u_brute || locate(x, body.mesh[u_brute]) ≈ locate(x, body.mesh[u_bvh])
                mesh_file == "sphere.stl" && @test sdf(body, x) ≈ √(x'x)-0.5f0  atol=15e-3
            end
        end
        # end
    end
end