module MeshingExt

using Meshing, GeometryBasics
import WaterLily: AbstractBody, measure, measure_sdf!
import WaterLilyMeshBodies: MeshBody

"""
    MeshBody(body::AbstractBody, N::NTuple{3}; t=0, mem=Array, boundary=true, kwargs...)

Marching-cubes mesh of the zero level set of `body` on the `N` grid, with the body velocity at the vertices.
"""
function MeshBody(body::AbstractBody, N::NTuple{3,Int}; t=0f0, mem=Array, boundary=true, kwargs...)
    σ = mem(zeros(Float32, N.+2)); measure_sdf!(σ, body, t)
    pts, fcs = Meshing.isosurface(Array(σ)[2:end-1,2:end-1,2:end-1], ntuple(i->(0.5,N[i]-0.5),3)...)
    b = MeshBody(GeometryBasics.Mesh(Point3f.(pts), GLTriangleFace.(fcs)); mem, boundary, kwargs...)
    copyto!(b.velocity, [hcat((measure(body,tri[:,i],t)[3] for i in 1:3)...) for tri in Array(b.mesh)]); b
end

end # module
