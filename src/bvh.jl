using StaticArrays
import ImplicitBVH
import ImplicitBVH: BoundingVolume, BBox, BSphere, memory_index, unsafe_isvirtual

# square distance to primitive
@fastmath @inline dist(x, b::BSphere) = max(√sum(abs2,x .- b.x) - b.r, 0)^2
@fastmath @inline function dist(x, b::BBox)
    s = zero(eltype(x))
    @inbounds for i in eachindex(b.up,b.lo)
        d = abs(x[i] - 0.5f0*(b.up[i] + b.lo[i])) - 0.5f0*(b.up[i] - b.lo[i])
        d>0 && (s+=d^2)
    end; s
end
@inline dist(x, b::BoundingVolume) = dist(x, b.volume)

# traverse the BVH
@inline function closest(x::SVector{D,T},bvh::ImplicitBVH.BVH,mesh;init_d²=floatmax(T),verbose=false) where {D,T}
    ncheck=lcheck=tcheck=Int32(0) # initialize counts
    best = (index=Int32(0),d²=init_d²,n=x,p=x) # initialize best element
    # Depth-First-Search
    tree = bvh.tree; length_nodes = length(bvh.nodes)
    i=Int32(1); while true
        @inbounds j = memory_index(tree,i)
        if j ≤ length_nodes # we are on a node
            verbose && (ncheck += 1)
            dist(x, @inbounds bvh.nodes[j]) < best.d² && (i = 2i; continue) # go deeper if closer than current best
        else # we reached a leaf
            verbose && (lcheck += 1)
            if dist(x, @inbounds bvh.leaves[j-length_nodes]) ≤ best.d²
                verbose && (tcheck += 1)
                @inbounds j = bvh.leaves[j-length_nodes].index # correct index in mesh
                tri = @inbounds mesh[j]
                p = locate(x,tri)
                d² = sum(abs2,x-p)
                if best.index == 0 || d² < best.d²
                    n = normal(tri)
                    best = (;index=Int32(j),d²,n,p)
                elseif d² ≈ best.d²
                    n = normal(tri)
                    abs((x-p)'n) > abs((x-best.p)'best.n) && (best = (;index=Int32(j),d²,n,p))
                end
            end
        end
        i = i>>trailing_ones(i)+1 # go to sibling, or uncle etc.
        (i==1 || unsafe_isvirtual(tree, i)) && break # search complete!
    end
    verbose && println("Checked $ncheck nodes, $lcheck leaves, $tcheck triangles")
    return best
end
