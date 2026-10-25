module MeshingExt

using Meshing, GeometryBasics
import WaterLily: AbstractBody, measure, measure_sdf!
import WaterLilyMeshBodies: MeshBody

"""
    MeshBody(body::AbstractBody, N::NTuple{3}; t=0f0, mem=Array, kwargs...)

Marching-cubes mesh of the zero level set of `body` on the `N` grid, with the body velocity at the vertices.
"""
function MeshBody(body::AbstractBody, N::NTuple{3,Int}; t=0f0, mem=Array, kwargs...)
    σ = mem(zeros(Float32, N.+2)); measure_sdf!(σ, body, t)
    pts, fcs = Meshing.isosurface(Array(σ)[2:end-1,2:end-1,2:end-1], ntuple(i->(0.5,N[i]-0.5),3)...)
    b = MeshBody(GeometryBasics.Mesh(Point3f.(pts), GLTriangleFace.(fcs)); mem, boundary=true, kwargs...)
    b.velocity .= (tri->hcat(ntuple(i->measure(body,tri[:,i],t)[3],3)...)).(b.mesh); b
end

end # module
