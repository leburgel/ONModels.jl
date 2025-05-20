using PEPSKit
using Zygote

# some utilities for PEPS fixed-point optimization using full rotation and reflection
# invariance

## Symmetrization

full_symmetrize!(psi::InfinitePEPS) = PEPSKit.symmetrize!(psi, PEPSKit.RotateReflect())

## Retraction and transport

function pepo_retract(x, η, α)
    # retract
    x´_partial, ξ = PEPSKit.peps_retract(x[1:2], η, α)
    x´ = (x´_partial..., deepcopy(x[3]))

    # symmetrize
    x´ = full_symmetrize!(x´[1]), x´[2], x´[3]
    ξ = full_symmetrize!(ξ)

    return x´, ξ
end

function pepo_transport!(ξ, x, η, α, x´)
    # transport
    PEPSKit.peps_transport!(ξ, x[1:2], η, α, x´[1:2])

    # symmetrize
    ξ = full_symmetrize!(ξ)

    return ξ
end

## Optimization

function generate_pepo_costfun(
    T::InfinitePEPO,
    boundary_alg=SimultaneousCTMRG(),
    rrule_alg=EigSolver(; iterscheme=:diffgauge),
)
    function pepo_costfun((psi, env2, env3)::Tuple{<:InfinitePEPS,<:CTMRGEnv,<:CTMRGEnv})
        E, gs = withgradient(psi) do ψ
            n2 = InfiniteSquareNetwork(ψ)
            env2′, info = PEPSKit.hook_pullback(
                leading_boundary, env2, n2, boundary_alg; alg_rrule=rrule_alg
            )
            n3 = InfiniteSquareNetwork(ψ, T)
            env3′, info = PEPSKit.hook_pullback(
                leading_boundary, env3, n3, boundary_alg; alg_rrule=rrule_alg
            )
            PEPSKit.ignore_derivatives() do
                PEPSKit.update!(env2, env2′)
                PEPSKit.update!(env3, env3′)
            end
            λ3 = network_value(n3, env3)
            λ2 = network_value(n2, env2)
            return -log(real(λ3 / λ2))
        end
        g = only(gs)
        g = full_symmetrize!(g)

        return E, g
    end
    return pepo_costfun
end
