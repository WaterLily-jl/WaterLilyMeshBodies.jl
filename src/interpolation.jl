"""

Constructs a `MotionInterpolation` object to interpolate mesh motion over time.

The data is assumed to be given as a 2D Array with dimensions `[N_snapshots × N_elements]`, where each element is a `SMatrix{3,3}`
representing the motion of a mesh element. Each snapshot corresponds to a specific time in the `times` vector. This time vector does
not need to be uniformly spaced. Interpolation is done through a non-uniform Catmull-Rom spline.

Periodic motion can be obtained by setting the `periodic` keyword argument to `true`. In that case, the motion will loop seamlessly
from the last snapshot back to the first.
"""
struct MotionInterpolation{T,M}
    motion_data   :: AbstractArray{M}
    velocity_data :: Union{Nothing, AbstractArray{M}}
    times         :: Vector{T}
    periodic      :: Bool
    period        :: T
    t             :: Ref{T}
    function MotionInterpolation(motion_data::AbstractArray{M}, velocity_data, times::Vector{T};
                                  periodic = false,
                                  period   = times[end]-times[1]+(times[end]-times[end-1])) where {T, M<:SMatrix{3,3,T}}
        new{T,M}(motion_data, velocity_data, times, periodic, T(period), Ref(zero(T)))
    end
end
function MotionInterpolation(motion_data::AbstractArray{M}, times::Vector{T}; kwargs...) where {T, M<:SMatrix{3,3,T}}
    MotionInterpolation(motion_data, nothing, times; kwargs...)
end

"""
    interpolate!(body::MeshBody{T}, a::MotionInterpolation, t)

Updates the body position and velocity using the MotionInterpolation at time `t=sum(sim.flow.Δt[1:end-1])`.

This function relies internally on `update!`, which requires assignment to the body to take full effects.
```julia
sim.body = interpolate!(sim.body, a, t)
```
"""
function interpolate!(body::MeshBody{T}, a::MotionInterpolation, t) where {T}
    dt = t - a.t[]; a.t[] = t
    k1, k2, k3, k4, τ, Δt_k, Δt_l, Δt_r = get_coeffs(a.times, t, a.period, Val(a.periodic))
    p0, p1 = a.motion_data[k1, :], a.motion_data[k2, :]
    p2, p3 = a.motion_data[k3, :], a.motion_data[k4, :]
    # non-uniform Catmull-Rom: Hermite basis with central-difference tangents
    m1 = @. (p2 - p0) / T(Δt_l)
    m2 = @. (p3 - p1) / T(Δt_r)
    h00, h10 =  2τ^3-3τ^2+1,  τ^3-2τ^2+τ
    h01, h11 = -2τ^3+3τ^2,    τ^3-τ^2
    motion_at_t = @. h00*p1 + (h10*T(Δt_k))*m1 + h01*p2 + (h11*T(Δt_k))*m2
    return update!(body, motion_at_t, dt)
end
interpolate!(body::AbstractBody,args...) = body
interpolate!(body::SetBody,args...) = SetBody(body.op,interpolate!(body.a,args...),interpolate!(body.b,args...))

# periodic: wrap t into one period, extend times modularly for the 4-point stencil
function get_coeffs(times, t, period, ::Val{true})
    N  = length(times)
    tl = times[1] + mod(t - times[1], period)
    k  = clamp(searchsortedfirst(times, tl) - 1, 1, N)
    # pt(i): periodic extension — wraps index and shifts by the correct number of periods
    pt(i) = times[mod1(i, N)] + fld(i-1, N) * period
    τ = (tl - pt(k)) / (pt(k+1) - pt(k))
    return mod1(k-1, N), mod1(k, N), mod1(k+1, N), mod1(k+2, N),
           τ, pt(k+1)-pt(k), pt(k+1)-pt(k-1), pt(k+2)-pt(k)
end
# non-periodic: clamp stencil to valid range
function get_coeffs(times, t, _, ::Val{false})
    k = clamp(searchsortedfirst(times, t) - 1, 2, length(times)-2)
    τ = (t - times[k]) / (times[k+1] - times[k])
    return k-1, k, k+1, k+2,
           τ, times[k+1]-times[k], times[k+1]-times[k-1], times[k+2]-times[k]
end

export MotionInterpolation, interpolate!