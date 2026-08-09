module FerriteShellsExt

using FerriteShells
using GeometryBasics
import WaterLilyMeshBodies: MeshBody, SetBody, save!, update!

# get nodes from the mesh, these are always the same
GeometryBasics.coordinates(grid::Grid{3,P,T}) where {P,T} = Point{3, T}[n.x.data for n in grid.nodes]

# convert Ferrite's native cell types into GeometryBasics faces, which are always flat and have 3 or 4 vertices
subfaces(c::Ferrite.Quadrilateral) = (c.nodes,)
# Q8 (serendipity)
subfaces(c::Ferrite.SerendipityQuadraticQuadrilateral) = (c.nodes[1:4],)
# Q9 split into 4 bilinear sub-quads
function subfaces(c::Ferrite.QuadraticQuadrilateral)
    n = c.nodes
    return ((n[1], n[5], n[9], n[8]),
             (n[5], n[2], n[6], n[9]),
             (n[9], n[6], n[3], n[7]),
             (n[8], n[9], n[7], n[4]))
end
# S3, easy
subfaces(c::Ferrite.Triangle) = (c.nodes,)
# S6 split into 4 sub-triangles
function subfaces(c::Ferrite.QuadraticTriangle)
    n = c.nodes
    return ((n[1], n[4], n[6]),
            (n[4], n[2], n[5]),
            (n[6], n[5], n[3]),
            (n[4], n[5], n[6]))
end

# decompose a Ferrite grid into a GeometryBasics mesh, which is what MeshBody expects
function GeometryBasics.decompose(::Type{F}, grid::Grid{3,P,T}) where {P,T,F<:AbstractFace}
    faces = F[]
    for c in grid.cells
        for f in subfaces(c) # in cases we have to decompose the mesh
            push!(faces, F(f))
        end
    end
    return faces
end

# do we have a tri or a quad?
facetype(c) = length(Ferrite.vertices(c)) == 3 ? TriangleFace{Int} : QuadFace{Int}

# convert a Ferrite grid into a MeshBody
function MeshBody(grid::Grid; kwargs...)
    F = facetype(first(grid.cells))
    points = GeometryBasics.decompose(Point{3, Float32}, grid)
    faces  = GeometryBasics.decompose(F, grid)
    MeshBody(GeometryBasics.Mesh(points, faces); kwargs...)
end

end # module
