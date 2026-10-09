using Logging
using WaterLily: pressure_force, viscous_force, pressure_moment

# count the full measurements logged by f()
struct FullCounter <: AbstractLogger; n::Base.RefValue{Int}; end
Logging.min_enabled_level(::FullCounter) = Logging.Debug
Logging.shouldlog(::FullCounter,args...) = true
Logging.catch_exceptions(::FullCounter) = false
Logging.handle_message(c::FullCounter,level,msg,args...;kw...) = (msg=="NarrowBand: full measurement" && (c.n[] += 1); nothing)
fulls(f) = (c = FullCounter(Ref(0)); with_logger(f,c); c.n[])

# diagonal displacements: realistic ½ cell steps, then 1 and 1.5 cells (the safety factor)
diagonal(D) = cumsum([fill(0.5f0,10); 1f0; 1.5f0]) .* Ref(fill(1/√Float32(D),SVector{D}))

@testset "NarrowBand closed MeshBody" begin
    # a bare closed MeshBody flood fills from scratch every measurement: the exact reference
    dims = (40,40,40); x₀ = SA{T}[16,16,16]
    sim(body) = Simulation(dims,(1,0,0),16;body,T)
    sphere() = MeshBody(joinpath(@__DIR__,"meshes","sphere.stl");scale=16f0,map=(x,t)->x.-16,boundary=true)
    a,b = sim(sphere()),sim(NarrowBand(sphere(),dims))
    b.flow.u .= a.flow.u .= rand(T,size(a.flow.u))
    for (n,Δx) in enumerate(diff([[zero(x₀)]; diagonal(3)]))
        shift(tri) = tri .+ Δx
        a.body = update!(a.body,shift.(a.body.mesh),1f0)
        b.body = update!(b.body,shift.(b.body.body.mesh),1f0)
        measure!(a,T(n)); @test fulls(()->measure!(b,T(n))) == 0
        b.flow.p .= a.flow.p .= rand(T,size(a.flow.p))
        @test a.flow.μ₀==b.flow.μ₀ && a.flow.μ₁==b.flow.μ₁ && a.flow.V==b.flow.V
        @test pressure_force(a)==pressure_force(b) && viscous_force(a)==viscous_force(b) &&
              pressure_moment(x₀,a)==pressure_moment(x₀,b)
    end
end

@testset "NarrowBand SetBody of closed MeshBodies" begin
    # compare to the union of its parts, each measured on its own
    dims,fd² = (56,40,40),6.25f0
    x₀ = SA{T}[16,16,16]; Δ = SA{T}[22,0,0] # second part offset
    sphere(Δ) = (b = MeshBody(joinpath(@__DIR__,"meshes","sphere.stl");scale=16f0,map=RigidMap(x₀,zero(x₀)),boundary=true);
                 update!(b,(tri->tri.+Δ).(b.mesh)))
    ball() = AutoBody((x,t)->√sum(abs2,x-Δ)-6,RigidMap(x₀,zero(x₀)))
    σ(body) = (d = zeros(T,dims.+2); measure_sdf!(d,body,0f0;fastd²=fd²); d)
    for parts in (()->(sphere(zero(Δ)),sphere(Δ)), ()->(sphere(zero(Δ)),ball()))
        a,b = parts(); band = NarrowBand(+(parts()...),dims)
        for (n,Δx) in enumerate([[zero(x₀)]; diagonal(3)])
            a,b,band = setmap.((a,b,band);x₀=x₀+Δx)
            σu,σr = zeros(T,dims.+2),min.(σ(a),σ(b))
            @test fulls(()->measure_sdf!(σu,band,0f0;fastd²=fd²)) == (n==1) # only the first is full
            # same sign everywhere, same distance in the band (≈: an AutoBody's sdf isn't normalized, a SetBody's is)
            @test all(I->signbit(σu[I])==signbit(σr[I]) && (σr[I]^2≥fd² || σu[I]≈σr[I]),inside(σu))
        end
    end
end
