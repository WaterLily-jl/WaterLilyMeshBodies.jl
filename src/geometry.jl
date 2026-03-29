# Geometric primitives and interpolation functions

using StaticArrays
using LinearAlgebra: cross
import WaterLily: ×

@fastmath @inline normal(tri::SMatrix) = hat(SVector(cross(tri[:,2]-tri[:,1],tri[:,3]-tri[:,1])))
@fastmath @inline hat(v) = v/(√(v'*v)+eps(eltype(v)))
@fastmath @inline center(tri::SMatrix) = SVector(sum(tri,dims=2)/3)

# linear shape function to interpolate inside element
@fastmath @inline shape_value(p::SVector{3,T},t) where T = SA{T}[√sum(abs2,×(t[:,2]-p,t[:,3]-p))
                                                                 √sum(abs2,×(p-t[:,1],t[:,3]-t[:,1]))
                                                                 √sum(abs2,×(t[:,2]-t[:,1],p-t[:,1]))]
@fastmath @inline get_velocity(p::SVector, tri, vel)= (dA=shape_value(p,tri); vel*dA/sum(dA))

# locate the closest point p to x on triangle tri
function locate(x::SVector{T},tri::SMatrix{T}) where T
    # unpack the triangle vertices
    a,b,c = tri[:,1],tri[:,2],tri[:,3]
    ab,ac,ap = b-a, c-a, x-a
    d1,d2 = ab'ap, ac'ap
    # is point `a` closest?
    ((d1 ≤ 0) && (d2 ≤ 0)) && (return a)
    # is point `b` closest?
    bp = x-b
    d3,d4 = ab'bp, ac'bp
    ((d3 ≥ 0) && (d4 ≤ d3)) && (return b)
    # is point `c` closest?
    cp = x-c
    d5,d6 = ab'cp, ac'cp
    ((d6 ≥ 0) && (d5 ≤ d6)) && (return c)
    # is segment 'ab' closest?
    vc = d1*d4 - d3*d2
    ((vc ≤ 0) && (d1 ≥ 0) && (d3 ≤ 0)) && (return a + (d1/(d1-d3))*ab)
    #  is segment 'ac' closest?
    vb = d5*d2 - d1*d6
    ((vb ≤ 0) && (d2 ≥ 0) && (d6 ≤ 0)) && (return a + (d2/(d2-d6))*ac)
    # is segment 'bc' closest?
    va = d3*d6 - d5*d4
    ((va ≤ 0) && (d4 ≥ d3) && (d5 ≥ d6)) && (return b + ((d4-d3)/(d4-d3+d5-d6))*(c-b))
    # closest is interior to `abc`
    denom = one(T) / (va + vb + vc)
    v,w= vb*denom,vc*denom
    return a + v*ab +w*ac
end
