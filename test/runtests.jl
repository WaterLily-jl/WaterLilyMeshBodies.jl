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

@testset "MeshBody base type" begin
    # the base type must follow the triangles, not the `scale` keyword, whose default
    # `1.f0` used to make every directly-constructed body a `MeshBody{Float32}`
    for S in (Float32, Float64)
        m = [SMatrix{3,3,S}(0,0,0, 1,0,0, 0,1,0)]
        body = MeshBody(m, zero(m), ImplicitBVH.BVH(BBox{S}.(m), BBox{S}))
        @test typeof(body).parameters[1] == S
        @test typeof(body.scale) == S && typeof(body.half_thk) == S
        @test eltype(SurfaceForces(body).pressure) == S
    end
    # scaling still works, and still sets the base type through the mesh it builds
    mesh = GeometryBasics.Mesh(Point{3,Float32}[(0,0,0),(1,0,0),(0,1,0)], [TriangleFace{Int}(1,2,3)])
    @test typeof(MeshBody(mesh; scale=2.f0)).parameters[1] == Float32
    @test only(MeshBody(mesh; scale=2.f0).mesh) ≈ 2only(MeshBody(mesh).mesh)
end

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

@testset "Non-affine map Jacobian" begin
    # folds z about the plane z=0.5, mirroring the symmetry maps used in production sims
    foldmap(x,t) = SA[x[1], x[2], abs(x[3]-0.5f0)]

    for mem in arrays
        mesh = mem([tri1])
        bvh = ImplicitBVH.BVH(BBox{T}.(mesh), BBox{T})
        body = MeshBody(mesh, zero(mesh), bvh, half_thk=0f0, map=foldmap)

        # move the triangle by +1 in z over dt=1 -> uniform real-space z-velocity of 1
        body = update!(body, mem([tri1 .+ SA{T}[0,0,1]]), 1.0)

        # query points symmetric about the fold plane, close to the triangle
        x_above = SA{T}[0.2,0.2,0.6] # physically above the fold plane
        x_below = SA{T}[0.2,0.2,0.4] # physically below the fold plane

        # a uniform real-space velocity must transform with opposite sign on either side of
        # a folding map's symmetry plane: this fails if the map's Jacobian is (incorrectly)
        # evaluated at the folded point ξ=map(x,t) instead of at the query point x, since both
        # x_above and x_below fold to the same ξ and would otherwise get the same Jacobian
        @test GPUArrays.@allowscalar measure(body, x_above, 0f0)[3][3] ≈ 1
        @test GPUArrays.@allowscalar measure(body, x_below, 0f0)[3][3] ≈ -1
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
        shift(tri) = tri .+ 0.5
        cache_body = update!(cache_body, shift.(cache_body.mesh), 1f0)
        abs_vel(vel) = maximum(√sum(abs2,vertex) for vertex in eachcol(vel))
        @test maximum(abs_vel.(cache_body.velocity)) < 1 # can't shift by more than 1 cell in one time step
        @test all((num_near,num_reached,num_farinside) .== count.(cache_body.cache))

        # warm-start should give same farinside count and same result after an integer shift
        measure_sdf!(σc, cache_body, 1f0; fastd²)
        cache_body = update!(cache_body, shift.(cache_body.mesh), 1f0)
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

@testset "MotionInterpolation" begin
    N = 4
    # 4 snapshots: shift all vertices uniformly by k-1 in each direction
    motion_data = [tri1 .+ T(k-1) for k in 1:N, j in 1:2]
    times_u  = T.(0:N-1)               # uniform spacing
    times_nu = T.([0, 0.3, 1.2, 3.0])  # non-uniform spacing

    for mem in arrays
        # helper: fresh body with original triangles
        mk_body() = (m = mem([tri1, R*tri1 .+ 1]);
                     MeshBody(m, zero(m), ImplicitBVH.BVH(BBox{T}.(m), BBox{T}), half_thk=0f0))

        # at a knot time τ=0, Hermite basis reduces to identity: returns exact snapshot
        b1 = interpolate!(mk_body(), MotionInterpolation(mem(motion_data), times_u), T(1))
        @test GPUArrays.@allowscalar all(b1.mesh[1] .≈ motion_data[2, 1])
        b2 = interpolate!(mk_body(), MotionInterpolation(mem(motion_data), times_u), T(2))
        @test GPUArrays.@allowscalar all(b2.mesh[1] .≈ motion_data[3, 1])

        # periodic: t=0 and t=period give the same mesh
        b_t0 = interpolate!(mk_body(), MotionInterpolation(mem(motion_data), times_u; periodic=true), T(0))
        b_tN = interpolate!(mk_body(), MotionInterpolation(mem(motion_data), times_u; periodic=true), T(N))
        @test GPUArrays.@allowscalar all(b_t0.mesh[1] .≈ b_tN.mesh[1])

        # non-uniform spacing: at a knot still returns exact snapshot
        b_nu = interpolate!(mk_body(), MotionInterpolation(mem(motion_data), times_nu), T(0.3))
        @test GPUArrays.@allowscalar all(b_nu.mesh[1] .≈ motion_data[2, 1])
    end
end

@testset "MeshBody nested ForwardDiff (GPU-safe)" begin
    using ForwardDiff
    function measure_sum(θ, mem)
        s, c = sincos(θ)
        Rmat = SA[c -s 0; s c 0; 0 0 1]
        body = MeshBody(joinpath(@__DIR__, "meshes", "sphere.stl");
                        scale=16f0, map=(x,_) -> Rmat*(x.-32), boundary=true, mem)
        sum(GPUArrays.@allowscalar WaterLily.measure(body, x, 0f0; fastd²=Inf32)[2][1]
            for x in (SA{Float32}[24,36,34], SA{Float32}[38,29,27], SA{Float32}[30,23,39]))
    end
    for f ∈ arrays, Tθ ∈ (Float32, Float64)
        cpu_d = ForwardDiff.derivative(t -> measure_sum(t, Array), Tθ(0.3))
        @test ForwardDiff.derivative(t -> measure_sum(t, f), Tθ(0.3)) ≈ cpu_d rtol=1e-3
    end
end

@testset "Force test" begin
    L = 64
    for f ∈ arrays
        # the STL geometry is centred on the origin: place it in the domain rather than with a
        # `map`, which the forces cannot apply. `∮(x₁+δn₁)n dA = V + δ∮n₁n dA` for `p=x₁`
        place(body) = update!(body, f([t .+ SA{T}[L,L,L] for t in Array(body.mesh)]), 0)
        force(body,δ) = (sim = Simulation((2L,2L,2L),(1,0,0),L;body,mem=f);
                         apply!(x->x[1], sim.flow.p);
                         WaterLily.pressure_force(SurfaceForces(sim.body), sim; δ))

        # a sphere of radius R, where ∮n₁n dA = A/3, up to the polyhedral error of the STL
        R = 0.9L
        sphere = place(MeshBody(joinpath(@__DIR__,"meshes/sphere.stl");
                                scale=T(1.8L), boundary=true, mem=f))
        for δ in (1f0,0.5f0)
            @test force(sphere,δ) ≈ (4/3*π*R^3 + δ*4π*R^2/3)*[1,0,0] rtol=2e-2
        end

        # a cube of side L, where the mesh is exact and so are both terms
        box = place(MeshBody(joinpath(@__DIR__,"meshes/box.stl");
                             scale=T(L/2), boundary=true, mem=f))
        for δ in (1f0,0.5f0)
            @test force(box,δ) ≈ (L^3 + 2δ*L^2)*[1,0,0] rtol=1e-5
        end
    end
end

@testset "Surface forces" begin
    L, N, δs = 20.0, 64, (1f0, 0.5f0)
    ν = 0.1f0 # must be non-zero, the viscous force is ν∫∂u/∂n dA
    # `box.stl` scaled by L/2 is exactly a cube of side L centred on the origin: shifted into
    # the domain it is a closed flat-faced body of volume L³ with faces of area L²
    cube(shift=32f0) = (b = MeshBody(joinpath(@__DIR__,"meshes/box.stl"); scale=T(L/2), boundary=true);
                        update!(b, [t .+ SA{T}[shift,shift,shift] for t in b.mesh], 0))
    mksim(body) = Simulation((N,N,N),(0,0,0),16; body, ν, T=Float32)

    sim = mksim(cube()); sf = SurfaceForces(sim.body)

    # a uniform pressure has no resultant on a closed body, ∮n dA = 0. This is the test that
    # catches a flipped or missing facet, both of which leave a net force behind
    apply!(x->3f0, sim.flow.p)
    for δ in δs
        @test all(abs.(WaterLily.pressure_force(sf, sim; δ)) .< 1e-3)
    end

    # a linear pressure gives the divergence theorem. `get_p` samples at c+δn, so for a
    # flat-faced body ∮(a⋅x + δ a⋅n) n dA = (V + 2δL²)a holds to machine precision
    for d in 1:3
        apply!(x->x[d], sim.flow.p)
        for δ in δs
            @test WaterLily.pressure_force(sf, sim; δ) ≈ (L^3+2δ*L^2)*[i==d for i in 1:3] rtol=1e-4
        end
    end

    # a plate of area A in the shear flow u = (γ(z-z₀),0,0) carries ν∫∂u/∂n dA = νγA
    A, γ, z₀ = 256f0, 0.01f0, 16f0
    pts = Point{3,Float32}[(8,8,z₀),(24,8,z₀),(24,24,z₀),(8,24,z₀)]
    plate = GeometryBasics.Mesh(pts, [TriangleFace{Int}(1,2,3), TriangleFace{Int}(1,3,4)])
    shear = mksim(MeshBody(plate; boundary=true))
    apply!((i,x)-> i==1 ? γ*(x[3]-z₀) : 0f0, shear.flow.u)
    sfs = SurfaceForces(shear.body)
    for δ in δs # the one-sided stencil is exact for a linear profile, at any δ
        @test WaterLily.viscous_force(sfs, shear; δ) ≈ [ν*γ*A,0,0] rtol=1e-3
    end

    # a thin shell in a uniform shear is dragged forwards on one face and backwards on the
    # other, so the two cancel. This is the test that catches a sign slip between the sides
    plate_shell = mksim(MeshBody(plate; boundary=false, half_thk=1f0))
    apply!((i,x)-> i==1 ? γ*(x[3]-z₀) : 0f0, plate_shell.flow.u)
    @test all(abs.(WaterLily.viscous_force(SurfaceForces(plate_shell.body), plate_shell; δ=1f0)) .< 1e-4)

    # a body translating with a uniform flow sees no relative motion, so it carries no shear.
    # This is the test that exercises the no-slip value `vₑ`, which the stencil needs to weigh
    # correctly: a uniform flow over a *stationary* wall does have a gradient, not zero shear
    moving = mksim(cube())
    apply!((i,x)-> i==1 ? 1f0 : 0f0, moving.flow.u)
    moving.body.velocity .= Ref(SA{Float32}[1 1 1; 0 0 0; 0 0 0]) # every vertex at (1,0,0)
    @test all(abs.(WaterLily.viscous_force(SurfaceForces(moving.body), moving; δ=1f0)) .< 1e-4)

    # the shear scales with the viscosity, and vanishes with it
    for factor in (2f0, 0f0)
        scaled = Simulation((N,N,N),(0,0,0),16; body=MeshBody(plate; boundary=true), ν=factor*ν, T=Float32)
        apply!((i,x)-> i==1 ? γ*(x[3]-z₀) : 0f0, scaled.flow.u)
        @test WaterLily.viscous_force(SurfaceForces(scaled.body), scaled; δ=1f0) ≈
              [factor*ν*γ*A,0,0] rtol=1e-3 atol=1e-8
    end
end