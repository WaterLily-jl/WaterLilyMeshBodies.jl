module WaterLilyMeshBodies

using WaterLily
import WaterLily: AbstractBody, SetBody, save!, update!
using FileIO, MeshIO
using ImplicitBVH, GeometryBasics
using WaterLilyNarrowBand

struct MeshBody{T,M,B,F} <: AbstractBody
    mesh::M
    velocity::M
    bvh::B
    map::F
    scale::T
    boundary::Bool
    half_thk::T
end
MeshBody(mesh::M,vel::M,bvh::B;map=(x,t)->x,scale=1.f0,boundary=false,half_thk=1.866f0) where {M,B} =
    MeshBody(mesh,vel,bvh,map,scale,boundary,half_thk)

# make it GPU compatible
using Adapt
Adapt.@adapt_structure MeshBody
Adapt.@adapt_structure SetBody

"""
    MeshBody(mesh::Union{Mesh, String};
             map::Function=(x,t)->x, boundary::Bool=false, half_thk::T=1.866f0,
             scale::T=1.f0, mem=Array, primitives::Union{BBox, BSphere}) where T

Constructor for a MeshBody:

  - `mesh::Union{Mesh, String}`: the GeometryBasics.Mesh or path to the mesh file to use to define the geometry.
  - `map(x::AbstractVector,t::Real)::AbstractVector`: coordinate mapping function.
  - `boundary::Bool=false`: whether the mesh is a boundary or not.
  - `half_thk::T=1.866f0`: half thickness to apply if the mesh is not a boundary, the type defines the base type of the MeshBody.
  - `scale::T=1.f0`: scale factor to apply to the mesh points, the type defines the base type of the MeshBody.
  - `mem=Array`: memory location. `Array`, `CuArray`, `ROCm` to run on CPU, NVIDIA, or AMD devices, respectively.
  - `primitive::Union{BBox, BSphere}=BBox`: bounding volume primitive to use in the ImplicitBVH.

If `boundary=true`, a flood-fill is used to determine the sign of the distance. If `boundary=false`, the sign is determined
by treating the mesh as a thin shell. Wrap a moving `MeshBody` in a `NarrowBand(body, dims)` to only measure it near its
surface, with a warm-started flood-fill.
"""
MeshBody(file_name::String; kwargs...) = MeshBody(load(file_name); kwargs...)
function MeshBody(mesh::Mesh; kwargs...)
    points = GeometryBasics.coordinates(mesh)
    faces  = GeometryBasics.faces(mesh)
    MeshBody(GeometryBasics.Mesh(points, GeometryBasics.decompose(GLTriangleFace, faces)); kwargs...)
end
function MeshBody(mesh::Mesh{3,T,P}; scale::T=1.f0, mem=Array, primitive=ImplicitBVH.BBox, kwargs...) where {T,P<:NgonFace{3}}
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
include("interpolation.jl")

export MeshBody, NarrowBand, save!, update!

end # module
