using Test
using Random
using ONModels
using TensorKit
using PEPSKit
using KrylovKit
using OptimKit

Random.seed!(123456)

## Setup

T = 3.0
convention = 2 # this one works best somehow
# should give f = -0.0882564910
f_ref = -0.0882564910

elt = ComplexF64
d = 3
n_max = 1
beta = 1 / T

symmetries = [Trivial] # TODO: test other symmetries

Vpepses = Dict(Trivial => ComplexSpace(3))
Venvs = Dict(Trivial => ComplexSpace(20))

boundary_alg = SimultaneousCTMRG(; maxiter=300, tol=1e-6, verbosity=2)
rrule_alg = EigSolver(;
    solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true), iterscheme=:diffgauge
)
ls_alg = HagerZhangLineSearch(; c₁=1e-4, c₂=1 - 1e-4, maxiter=10, maxfg=10)
optimization_alg = LBFGS(32; maxiter=35, gradtol=1e-5, verbosity=3, linesearch=ls_alg)

@testset "$(d)D classical XY model for $symmetry symmetry" for symmetry in symmetries
    O, E = ONModels.classical_XY(elt, symmetry, d; n_max, beta, convention)
    if convention == 2
        O = flip(O, [1, 5, 6])
    end

    Vpeps = Vpepses[symmetry]
    Venv = Venvs[symmetry]

    pepo = InfinitePEPO(O; unitcell=(1, 1, 1))
    psi0 = PEPSKit.peps_normalize(initializePEPS(pepo, Vpeps))
    env2_0 = CTMRGEnv(InfiniteSquareNetwork(psi0), Venv)
    env3_0 = CTMRGEnv(InfiniteSquareNetwork(psi0, pepo), Venv)

    pepo_costfun = generate_pepo_costfun(pepo, boundary_alg, rrule_alg)

    (psi_final, env2_final, env3_final), f, = optimize(
        pepo_costfun,
        (psi0, env2_0, env3_0),
        optimization_alg;
        inner=PEPSKit.real_inner,
        retract=pepo_retract,
        (transport!)=(pepo_transport!),
    )

    @test f ≈ f_ref rtol = 1e-2
end
