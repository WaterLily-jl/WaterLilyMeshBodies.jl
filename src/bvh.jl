# BVH traversal and bounding volume functions

using StaticArrays
import ImplicitBVH
import ImplicitBVH: BoundingVolume, BBox, BSphere, memory_index, unsafe_isvirtual

@fastmath @inline inside(x::SVector, b::BoundingVolume) = inside(x, b.volume)
@fastmath @inline inside(x::SVector, b::BBox) = all(b.lo.-4 .≤ x) && all(x .≤ b.up.+4)
@fastmath @inline inside(x::SVector, b::BSphere) = √sum(abs2,x .- b.x) - b.r ≤ 4

# compute the square distance to primitive
dist(x, b::BSphere) = max(√sum(abs2,x .- b.x) - b.r, 0)^2
function dist(x, b::BBox)
    c = (b.up .+ b.lo) ./ 2
    r = (b.up .- b.lo) ./ 2
    sum(abs2, max.(abs.(x .- c) .- r, 0))
end
dist(x, b::BoundingVolume) = dist(x, b.volume)

# traverse the BVH
@inline function closest(x::SVector{D,T},bvh::ImplicitBVH.BVH,mesh;a=floatmax(T),verbose=false) where {D,T}
    tree = bvh.tree; length_nodes = length(bvh.nodes)
    u=ncheck=lcheck=tcheck=Int32(0) # initialize
    # Depth-First-Search
    i=1; while true
        @inbounds j = memory_index(tree,i)
        if j ≤ length_nodes # we are on a node
            verbose && (ncheck += 1)
            dist(x, bvh.nodes[j]) < a && (i = 2i; continue) # go deeper if closer than current best
        else # we reached a leaf
            verbose && (lcheck += 1)
            if dist(x, bvh.leaves[j-length_nodes]) < a
                verbose && (tcheck += 1)
                @inbounds j = bvh.leaves[j-length_nodes].index # correct index in mesh
                d = d²_fast(x, mesh[j])
                d<a && (a=d; u=Int32(j))  # Replace current best
            end
        end
        i = i>>trailing_ones(i)+1 # go to sibling, or uncle etc.
        (i==1 || unsafe_isvirtual(tree, i)) && break # search complete!
    end
    verbose && println("Checked $ncheck nodes, $lcheck leaves, $tcheck triangles")
    return a,u
end
