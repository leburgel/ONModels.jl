using Test
using ONModels
using TensorKit
using MPSKit

T = 0.5
# should give λ = 5.8744125865, f = -0.885303034876538, E = -0.588527079168918
f_ref = -0.885303034876538
e_ref = -0.588527079168918

elt = ComplexF64
d = 2
l_max = 6
beta = 1 / T
convention = 1

symmetries = [SU2Irrep, Trivial] # TODO: fix and test O3Irrep symmetry

Vspaces = Dict(
    SU2Irrep => SU2Space(0 => 5, 1 => 2, 2 => 4, 3 => 2, 4 => 3, 5 => 1, 6 => 2, 7 => 1),
    SU2Irrep ⊠ Z2Irrep => Vect[SU2Irrep ⊠ Z2Irrep](
        (0, 0) => 5,
        (1, 1) => 2,
        (2, 0) => 4,
        (3, 1) => 2,
        (4, 0) => 3,
        (5, 1) => 1,
        (6, 0) => 2,
        (7, 1) => 1,
    ),
    Trivial => ComplexSpace(24),
)

boundary_alg = VUMPS(; tol=1e-8, verbosity=3, maxiter=100)

@testset "$(d)D classical RP^2 model for $symmetry symmetry" for symmetry in symmetries
    O, E = ONModels.classical_RP2(elt, symmetry, d; l_max, beta, convention)
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

    @test f ≈ f_ref rtol = 1e-3
    @test e ≈ e_ref rtol = 1e-3
end
