# =================================================================================================
# Diffusion kurtosis along one direction, or of direction-averaged data
# =================================================================================================

"""
    ADCKurtosis(; b)

Diffusion signal with kurtosis, along one gradient direction or for direction-averaged
("powder-averaged") data (Jensen et al., Magn Reson Med 2005;53:1432):

    S = S0·exp(-b·ADC + b²·ADC²·K/6)

K measures how far diffusion departs from a Gaussian; with K = 0 this is [`ADCDecay`](@ref).

- acquisition: `b` (s/mm²), one per observation; K needs b-values up to about 2000–3000 s/mm²
- free by default: `S0` (the signal at b = 0), `ADC` (mm²/s; the apparent diffusion
  coefficient, D_app in kurtosis papers), `K` (apparent kurtosis, no units; around 1 in brain
  tissue)

Bounds: `K` from 0 (any mix of Gaussian compartments gives K ≥ 0) to 3, a plausibility limit;
widen it with `fit(m, data; K = Free(upper = 10))`. `ADC` from 0 to 4e-3 mm²/s (above free
water, as in [`ADCDecay`](@ref); for b in s/mm², `ADC = Free(upper = Inf)` removes it). The modelled signal rises again
at high b once b·ADC·K > 3. That limit is not enforced (it involves two parameters, so it is not
a bound), so check `b_max·ADC·K < 3` on the results.

Starts at `ADC = 1/mean(b)` and `K = 1`.

```julia
m = ADCKurtosis(b = [0, 500, 1000, 1500, 2000, 2500])
```
"""
@kwdef struct ADCKurtosis{A<:AbstractVector} <: MRIModel
    b::A                                # b-values (s/mm²)
end

parameters(::Type{<:ADCKurtosis}) = (
    S0 = Free(start = :scale, lower = 0),
    ADC = Free(start = m -> 1 / mean(m.b), lower = 0, upper = _DMAX),   # start from the b-values
    K = Free(start = 1.0, lower = 0, upper = 3),
)

function signal!(out, m::ADCKurtosis, p)
    (; b) = m
    (; S0, ADC, K) = p
    @inbounds for j in eachindex(out, b)
        bD = b[j] * ADC
        out[j] = S0 * exp(-bD + bD^2 * K / 6)
    end
    return out
end
