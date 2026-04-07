module WaterLilyMeshBodies

using WaterLily
import WaterLily: AbstractBody, SetBody, save!, update!
using FileIO, MeshIO
using ImplicitBVH, GeometryBasics

struct MeshBody{T,M,B,F,C} <: AbstractBody
    mesh::M
    velocity::M
    bvh::B
    map::F
    scale::T
    boundary::Bool
    half_thk::T
    cache::C
end
function MeshBody(mesh::M,vel::M,bvh::B;map=(x,t)->x,scale=1.f0,boundary=false,half_thk=1.866f0,size=nothing) where {M,B}
    cache = isnothing(size) ? nothing : ntuple(i -> similar(mesh, Bool, size .+ 2), 3)
    MeshBody{eltype(scale),M,B,typeof(map),typeof(cache)}(mesh,vel,bvh,map,scale,boundary,half_thk,cache)
end
using Adapt
# make it GPU compatible
function Adapt.adapt_structure(to, body::MeshBody)
    mesh = Adapt.adapt(to, body.mesh)
    velocity = Adapt.adapt(to, body.velocity)
    bvh = Adapt.adapt(to, body.bvh)
    cache = Adapt.adapt(to, body.cache)
    MeshBody{typeof(body.scale),typeof(mesh),typeof(bvh),typeof(body.map),typeof(cache)}(
        mesh, velocity, bvh, body.map, body.scale, body.boundary, body.half_thk, cache)
end

"""
    MeshBody(mesh::Union{Mesh, String};
             map::Function=(x,t)->x, boundary::Bool=false, half_thk::T=1.866f0,
             scale::T=1.f0, mem=Array, primitives::Union{BBox, BSphere}) where T

Constructor for a MeshBody:

  - `mesh::Union{Mesh, String}`: the GeometryBasics.Mesh or path to the mesh file to use to define the geometry.
  - `map(x::AbstractVector,t::Real)::AbstractVector`: coordinate mapping function.
  - `boundary::Bool`: whether the mesh is a boundary or not.
  - `half_thk::T`: half thickness to apply if the mesh is not a boundary, the type defines the base type of the MeshBody, default is Float32.
  - `scale::T`: scale factor to apply to the mesh points, the type defines the base type of the MeshBody, default is Float32.
  - `mem`: memory location. `Array`, `CuArray`, `ROCm` to run on CPU, NVIDIA, or AMD devices, respectively.
  - `primitive::Union{BBox, BSphere}`: bounding volume primitive to use in the ImplicitBVH. Default is Axis-Aligned Bounding Box.

"""
MeshBody(file_name::String; kwargs...) = MeshBody(load(file_name); kwargs...)
function MeshBody(mesh::Mesh; scale::T=1.f0, mem=Array, primitive=ImplicitBVH.BBox, kwargs...) where T
    # device array of the mesh that we store
    mesh = [hcat([mesh[i]...]...)*T(scale) for i in 1:length(mesh)] |> mem
    # make the BVH
    bvh = BVH(primitive{T}.(mesh), primitive{T})
    # make the mesh and return
    MeshBody(mesh, zero(mesh), bvh; scale=T(scale), kwargs...)
end

include("geometry.jl")
include("bvh.jl")
include("measure.jl")
include("update.jl")
include("io.jl")

export MeshBody, save!, update!

end # module