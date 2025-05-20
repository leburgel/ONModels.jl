using Test
using ONModels
using TensorKit
using MPSKit

T = 1.0
# should give λ = 1.8045597575, f = -0.5903166603879662, E = -0.659106277754763
f_ref = -0.5903166603879662
e_ref = -0.659106277754763

elt = ComplexF64
d = 2
n_max = 5
beta = 1 / T
convention = 1

symmetries = [U1Irrep, Trivial] # TODO: fix and test CU1Irrep symmetry

Vspaces = Dict(
    U1Irrep => U1Space(
        0 => 11,
        1 => 11,
        -1 => 11,
        2 => 8,
        -2 => 8,
        3 => 6,
        -3 => 6,
        4 => 2,
        -4 => 2,
        5 => 1,
        -5 => 1,
    ),
    CU1Irrep => CU1Space(
        (0, 0) => 5, (1, 2) => 6, (2, 2) => 6, (3, 2) => 8, (4, 2) => 8, (5, 2) => 6
    ),
    Trivial => ComplexSpace(25),
)

boundary_alg = VUMPS(; tol=1e-8, verbosity=3, maxiter=100)

@testset "$(d)D classical XY model for $symmetry symmetry" for symmetry in symmetries
    O, E = ONModels.classical_XY(elt, symmetry, d; n_max, beta, convention)
    T = InfiniteMPO([O])
    P = physicalspace(T, 1)
    V = Vspaces[symmetry]

    psi0 = InfiniteMPS(randn, ComplexF64, [P], [V])

    psi, env, err = leading_boundary(psi0, T, boundary_alg)

    λ = expectation_value(psi, T, env)
    abs(imag(λ)) > 1e-12 && @warn "Oops, imaginary!"
    λ = real(λ)

    # free energy
    f = -log(λ) / beta

    # energy per link
    num = MPSKit.contract_mpo_expval(psi.AC[1], env.GLs[1], E, env.GRs[1], psi.AC[1])
    denom = MPSKit.contract_mpo_expval(psi.AC[1], env.GLs[1], O, env.GRs[1], psi.AC[1])
    e = real(num / denom) / 2

    @test f ≈ f_ref rtol = 1e-4
    @test e ≈ e_ref rtol = 1e-3
end
