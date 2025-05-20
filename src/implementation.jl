# actually do the thing

# Auxiliary functions
# -------------------

const PFTensor{S} = AbstractTensorMap{<:Any,S,2,2}
const PEPSTensor{S} = AbstractTensorMap{<:Any,S,1,4}
const PEPOTensor{S} = AbstractTensorMap{<:Any,S,2,4}

const O3Irrep = SU2Irrep ⊠ Z2Irrep

_physical_charge(::Type{SU2Irrep}, l::SU2Irrep) = l
_physical_charge(::Type{O3Irrep}, l::SU2Irrep) = l ⊠ Z2Irrep(convert(Int, mod(l.j, 2)))
_physical_charge(::Type{U1Irrep}, n::U1Irrep) = n
function _physical_charge(::Type{CU1Irrep}, n::U1Irrep)
    @show n.charge
    return CU1Irrep(n.charge)
end

## tensor space conventions

tensor_space(::Val{2}, args...) = mpo_space(args...)
tensor_space(::Val{3}, args...) = pepo_space(args...)

mpo_space(::Val{1}, V::ElementarySpace) = V ⊗ V ← V ⊗ V
mpo_space(::Val{2}, V::ElementarySpace) = V' ⊗ V' ← V ⊗ V
mpo_space(::Val{3}, V::ElementarySpace) = V ⊗ V ← V' ⊗ V'

pepo_space(::Val{1}, V::ElementarySpace) = V ⊗ V' ← V ⊗ V ⊗ V' ⊗ V'
pepo_space(::Val{2}, V::ElementarySpace) = V' ⊗ V' ← V ⊗ V ⊗ V ⊗ V
pepo_space(::Val{3}, V::ElementarySpace) = V ⊗ V ← V' ⊗ V' ⊗ V' ⊗ V'

## Trivial symmetry conversion

function _to_trivial_space(V::ElementarySpace)
    trivial_V = ComplexSpace(dim(V))
    return isdual(V) ? trivial_V' : trivial_V
end

function convert_symmetry(
    ::Type{S}, O::AbstractTensorMap{<:Any,E}
) where {S<:Sector,E<:ElementarySpace}
    @assert sectortype(E) == S "Can only do trivial conversions."
    return O
end
function convert_symmetry(::Type{Trivial}, O::AbstractTensorMap)
    Oarr = convert(Array, O)
    trivial_tensorspace =
        prod(_to_trivial_space, codomain(O)) ← prod(_to_trivial_space, domain(O))
    return TensorMap(Oarr, trivial_tensorspace)
end

## Energy tensor from partition function tensor

energy_tensor(::Val{2}, args...) = energy_mpo(args...)
energy_tensor(::Val{3}, args...) = energy_pepo(args...)

function energy_mpo(
    O::PFTensor, energy_weights::Dict{S,<:Number}, interaction_weights::Dict{S,<:Number}
) where {S<:Sector}
    E_horizontal = copy(O)
    E_vertical = copy(O)
    for (s, f) in fusiontrees(O)
        abs(only(O[s, f])) > eps() || continue
        E_horizontal[s, f] *=
            energy_weights[f.uncoupled[2]] / interaction_weights[f.uncoupled[2]]
        E_vertical[s, f] *=
            energy_weights[f.uncoupled[1]] / interaction_weights[f.uncoupled[1]]
    end
    return E_horizontal + E_vertical
end

function energy_pepo(
    O::PEPOTensor, energy_weights::Dict{S,<:Number}, interaction_weights::Dict{S,<:Number}
) where {S<:Sector}
    E_x = copy(O)
    E_y = copy(O)
    E_z = copy(O)
    for (s, f) in fusiontrees(O)
        norm(O[s, f]) > eps() || continue
        E_x[s, f] *= energy_weights[f.uncoupled[2]] / interaction_weights[f.uncoupled[2]]
        E_y[s, f] *= energy_weights[f.uncoupled[1]] / interaction_weights[f.uncoupled[1]]
        E_z[s, f] *= energy_weights[s.uncoupled[2]] / interaction_weights[s.uncoupled[2]]
    end
    return E_x + E_y + E_z
end

## O3

function edge_coefficient(l::SU2Irrep, f::Function)
    integrand(x) = legendre(x, l.j) * f(x)
    return 2 * pi * quadgk(integrand, -1, 1)[1]
end
edge_coefficient(l::O3Irrep, f::Function) = edge_coefficient(l[1], f)

function ON_vertex((l1, l2, l3)::NTuple{3,SU2Irrep}, (a1, a2, a3)::NTuple{3,Bool})
    # do the shifty thing?
    arrows = (-1) .^ (a1, a2, a3)
    if all(arrows .== arrows[1])
        shift = 0
    else
        shift = 3 - findfirst(arrows .== -sum(arrows))
    end

    l1, l2, l3 = circshift([l1, l2, l3], shift)
    a1, a2, a3 = circshift([a1, a2, a3], shift)

    @assert a1 == a2 "Nope"

    vertex_factor =
        sqrt(dim(l1) * dim(l2) / (4 * pi)) *
        (-1)^(l1.j - l2.j - !a3 * l3.j) * # toggle extra sign based on coupled arrow?
        wigner3j(l1.j, l2.j, l3.j, 0, 0, 0)

    return vertex_factor
end
function ON_vertex((l1, l2, l3)::NTuple{3,O3Irrep}, (a1, a2, a3)::NTuple{3,Bool})
    return ON_vertex((l1[1], l2[1], l3[1]), (a1, a2, a3))
end

ON_tree_normalization(::Type{SU2Irrep}) = 1 / (4 * pi)
ON_tree_normalization(::Type{O3Irrep}) = 1 / (4 * pi)

## O2

function edge_coefficient(n::U1Irrep, f::Function)
    integrand(x) = exp(-im * n.charge * x) * f(x)
    return quadgk(integrand, 0, 2 * pi)[1] / (2 * pi)
end
function edge_coefficient(n::CU1Irrep, f::Function)
    integrand(x) = exp(-im * n.j * x) * f(x)
    return quadgk(integrand, 0, 2 * pi)[1] / (2 * pi)
end

function ON_vertex((n1, n2, n3)::NTuple{3,U1Irrep}, (a1, a2, a3)::NTuple{3,Bool})
    # arrows don't matter at all here
    vertex_factor = n1.charge + n2.charge == n3.charge
    vertex_factor || @warn "Vertex factor is zero!"

    return 1.0 * vertex_factor
end
function ON_vertex((l1, l2, l3)::NTuple{3,CU1Irrep}, (a1, a2, a3)::NTuple{3,Bool})
    @assert false "Nope"
end

ON_tree_normalization(::Type{U1Irrep}) = 1.0
ON_tree_normalization(::Type{CU1Irrep}) = 1.0

# ON

function ON_tree_factor(s::FusionTree{S}, f::FusionTree{S}) where {S<:Sector}
    return ON_tree_factor(s; split=true) *
           ON_tree_factor(f; split=false) *
           ON_tree_normalization(S)
end
function ON_tree_factor(f::FusionTree{S}; split::Bool=false) where {S<:Sector}
    dual2arrow = Base.Fix2(xor, split)
    fact = 1.0
    # go through the entire tree
    l1 = f.uncoupled[1]
    l2 = f.uncoupled[2]
    a1 = dual2arrow(f.isdual[1])
    a2 = dual2arrow(f.isdual[2])
    for (i, l3) in enumerate(f.innerlines)
        a3 = dual2arrow(true) # outgoing
        fact *= ON_vertex((l1, l2, l3), (a1, a2, a3))
        l1 = l3
        l2 = f.uncoupled[2 + i]
        a1 = dual2arrow(false) # incoming
        a2 = f.isdual[2 + i]
    end
    # don't forget the root
    fact *= ON_vertex((l1, l2, f.coupled), (a1, a2, dual2arrow(true)))
    return fact
end

function ON_tensor(
    elt::Type{<:Number},
    symmetry::Type{<:Sector},
    d::Int,
    interaction_weights::Dict{<:Sector,<:Number};
    convention::Int=1,
)
    V = Vect[symmetry](l => 1 for l in keys(interaction_weights))

    O = zeros(elt, tensor_space(Val(d), Val(convention), V))

    for (s, f) in fusiontrees(O)
        O[s, f] .=
            sqrt(
                prod(interaction_weights[l] for l in s.uncoupled) *
                prod(interaction_weights[l] for l in f.uncoupled),
            ) * ON_tree_factor(s, f)
    end

    return O
end

## O2

# Heisenberg model
# ----------------

"""
    classical_heisenberg(
        elt::Type{<:Number}=ComplexF64,
        symmetry::Type{Trivial}=Trivial,
        d::Int=2;
        beta::Float64=0.8,
        l_max::Int=1,
        convention::Int=1,
    )

Local tensor corresponding to the partition function of the `d`-dimensional classical
Heisenberg model, defined as

```math
\\mathcal{Z}(\\beta) = \\sum_{\\{s\\}} \\exp(-\\beta H(s)) \\text{ with } H(s) = -\\sum_{\\langle i, j \\rangle} \\vec{s}_i \\vec{s}_j
```
where ``\\vec{s}_i`` denotes a three-component classical spin of unit length at site ``i``
of the `d`-dimensional hypercubic lattice.
"""
function classical_heisenberg end
function classical_heisenberg(symmetry::Type{<:Sector}; kwargs...)
    return classical_heisenberg(ComplexF64, symmetry, 2; kwargs...)
end
function classical_heisenberg(
    elt::Type{<:Number}=ComplexF64,
    symmetry::Type{<:Sector}=SU2Irrep,
    d::Int=2;
    beta::Float64=0.8,
    l_max::Int=1,
    convention::Int=1,
)
    construction_symmetry = symmetry == Trivial ? SU2Irrep : symmetry

    # get interaction weights
    interaction_factor(x) = exp(beta * x)
    interaction_weights = Dict(
        _physical_charge(construction_symmetry, SU2Irrep(l)) =>
            edge_coefficient(SU2Irrep(l), interaction_factor) for l in 0:l_max
    )

    # use these to build the partition function tensor
    O = ON_tensor(elt, construction_symmetry, d, interaction_weights; convention)

    # also construct the local energy tensor while we're at it
    energy_factor(x) = -x * exp(beta * x)
    energy_weights = Dict(
        _physical_charge(construction_symmetry, SU2Irrep(l)) =>
            edge_coefficient(SU2Irrep(l), energy_factor) for l in 0:l_max
    )
    E = energy_tensor(Val(d), O, energy_weights, interaction_weights)

    # convert if necessary
    O = convert_symmetry(symmetry, O)
    E = convert_symmetry(symmetry, E)

    return O, E
end

# RP2
# ---

"""
    classical_RP2(
        elt::Type{<:Number}=ComplexF64,
        symmetry::Type{Trivial}=Trivial,
        d::Int=2;
        beta::Float64=0.8,
        l_max::Int=2,
        convention::Int=1,
    )

Local tensor corresponding to the partition function of the `d`-dimensional classical
``\\mathrm{RP}^{2}`` model, defined as

```math
\\mathcal{Z}(\\beta) = \\sum_{\\{s\\}} \\exp(-\\beta H(s)) \\text{ with } H(s) = -\\sum_{\\langle i, j \\rangle} \\left ( \\vec{s}_i \\cdot \\vec{s}_j \\right )^2
```
where ``\\vec{s}_i`` denotes a three-component classical spin of unit length at site ``i``
of the `d`-dimensional hypercubic lattice.
"""
function classical_RP2 end
function classical_RP2(symmetry::Type{<:Sector}; kwargs...)
    return classical_RP2(ComplexF64, symmetry, 2; kwargs...)
end
function classical_RP2(
    elt::Type{<:Number}=ComplexF64,
    symmetry::Type{<:Sector}=SU2Irrep,
    d::Int=2;
    beta::Float64=0.8,
    l_max::Int=2,
    convention::Int=1,
)
    construction_symmetry = symmetry == Trivial ? SU2Irrep : symmetry

    # get interaction weights
    interaction_factor(x) = exp(beta * x^2)
    interaction_weights = Dict(
        _physical_charge(construction_symmetry, SU2Irrep(l)) =>
            edge_coefficient(SU2Irrep(l), interaction_factor) for l in 0:2:l_max
    )

    # use these to build the partition function tensor
    O = ON_tensor(elt, construction_symmetry, d, interaction_weights; convention)

    # also construct the local energy tensor while we're at it
    energy_factor(x) = -x^2 * exp(beta * x^2)
    energy_weights = Dict(
        _physical_charge(construction_symmetry, SU2Irrep(l)) =>
            edge_coefficient(SU2Irrep(l), energy_factor) for l in 0:2:l_max
    )
    energy_weights = filter(kv -> abs(last(kv)) > 1e-12, energy_weights)
    E = energy_tensor(Val(d), O, energy_weights, interaction_weights)

    # convert if necessary
    O = convert_symmetry(symmetry, O)
    E = convert_symmetry(symmetry, E)

    return O, E
end

"""
    classical_heisenbergRP2(
        elt::Type{<:Number}=ComplexF64,
        symmetry::Type{Trivial}=Trivial,
        d::Int=2;
        beta::Float64=0.8,
        theta::Float64=0.0,
        l_max::Int=2,
        convention::Int=1,
    )

Local tensor corresponding to the partition function of the `d`-dimensional classical
Heisenberg-``\\mathrm{RP}^{2}`` model, defined as

```math
\\mathcal{Z}(\\beta) = \\sum_{\\{s\\}} \\exp(-\\beta H(s)) \\text{ with } H(s) = -\\sum_{\\langle i, j \\rangle} \\left ( \\cos(\\theta) \\vec{s}_i \\cdot \\vec{s}_j + \\sin(\\theta) \\left ( \\vec{s}_i \\cdot \\vec{s}_j \\right )^2 \\right )
```
where ``\\vec{s}_i`` denotes a three-component classical spin of unit length at site ``i``
of the `d`-dimensional hypercubic lattice.
"""
function classical_heisenbergRP2 end
function classical_heisenbergRP2(symmetry::Type{<:Sector}; kwargs...)
    return classical_heisenbergRP2(ComplexF64, symmetry, 2; kwargs...)
end
function classical_heisenbergRP2(
    elt::Type{<:Number}=ComplexF64,
    symmetry::Type{<:Sector}=SU2Irrep,
    d::Int=2;
    beta::Float64=0.8,
    theta::Float64=0.0,
    l_max::Int=1,
    convention::Int=1,
)
    construction_symmetry = symmetry == Trivial ? SU2Irrep : symmetry

    # get interaction weights
    interaction_factor(x) = exp(beta * (cos(theta) * x + sin(theta) * x^2))
    interaction_weights = Dict(
        _physical_charge(construction_symmetry, SU2Irrep(l)) =>
            edge_coefficient(SU2Irrep(l), interaction_factor) for l in 0:l_max
    )
    interaction_weights = filter(kv -> abs(last(kv)) > 1e-12, interaction_weights)

    # use these to build the partition function tensor
    O = ON_tensor(elt, construction_symmetry, d, interaction_weights; convention)

    # also construct the local energy tensor while we're at it
    function energy_factor(x)
        return -(cos(theta) * x + sin(theta) * x^2) *
               exp(beta * (cos(theta) * x + sin(theta) * x^2))
    end
    energy_weights = Dict(
        _physical_charge(construction_symmetry, SU2Irrep(l)) =>
            edge_coefficient(SU2Irrep(l), energy_factor) for l in 0:2:l_max
    )
    energy_weights = filter(kv -> abs(last(kv)) > 1e-12, energy_weights)
    E = energy_tensor(Val(d), O, energy_weights, interaction_weights)

    # convert if necessary
    O = convert_symmetry(symmetry, O)
    E = convert_symmetry(symmetry, E)

    return O, E
end

"""
    classical_gaugeRP2(
        elt::Type{<:Number}=ComplexF64,
        symmetry::Type{Trivial}=Trivial,
        d::Int=2;
        beta::Float64=0.8,
        theta::Float64=0.0,
        l_max::Int=2,
        convention::Int=1,
    )

Local tensor corresponding to the partition function of the `d`-dimensional classical
gauge-``\\mathrm{RP}^{2}`` model, defined as

```math
\\mathcal{Z}(\\beta) = \\sum_{\\{s\\}} \\exp(-\\beta H(s)) \\text{ with } H(s) = -\\sum_{\\langle i, j \\rangle} \\left ( \\cos(\\theta) \\vec{s}_i \\cdot \\vec{s}_j + \\sin(\\theta) \\left ( \\vec{s}_i \\cdot \\vec{s}_j \\right )^2 \\right )
```
where ``\\vec{s}_i`` denotes a three-component classical spin of unit length at site ``i``
of the `d`-dimensional hypercubic lattice.
"""
function classical_gaugeRP2 end
# TODO: implement

# XY
# ---

"""
    classical_XY(
        elt::Type{<:Number}=ComplexF64,
        symmetry::Type{Trivial}=Trivial,
        d::Int=2;
        l_max::Int=1,
        beta::Float64=0.8,
    )

Local tensor corresponding to the partition function of the `d`-dimensional classical
``\\mathrm{XY}`` model, defined as

```math
\\mathcal{Z}(\\beta) = \\sum_{\\{s\\}} \\exp(-\\beta H(s)) \\text{ with } H(s) = -\\sum_{\\langle i, j \\rangle} \\left ( \\vec{s}_i \\cdot \\vec{s}_j \\right )^2
```
where ``\\vec{s}_i`` denotes a two-component classical spin of unit length at site ``i``
of the `d`-dimensional hypercubic lattice.
"""
function classical_XY end
function classical_XY(symmetry::Type{<:Sector}; kwargs...)
    return classical_XY(ComplexF64, symmetry, 2; kwargs...)
end
function classical_XY(
    elt::Type{<:Number}=ComplexF64,
    symmetry::Type{<:Sector}=U1Irrep,
    d::Int=2;
    beta::Float64=0.8,
    n_max::Int=1,
    convention::Int=1,
)
    symmetry == CU1Irrep && throw(ArgumentError("CU1Irrep symmetry not implemented yet"))
    construction_symmetry = symmetry == Trivial ? U1Irrep : symmetry

    # get interaction weights; they're just Bessel functions in this case
    interaction_weights = Dict(
        _physical_charge(construction_symmetry, U1Irrep(n)) => besseli(n, beta) for
        n in (-n_max):n_max
    )

    # use these to build the partition function tensor
    O = ON_tensor(elt, construction_symmetry, d, interaction_weights; convention)

    # also construct the local energy tensor while we're at it
    energy_factor(x) = -cos(x) * exp(beta * cos(x))
    energy_weights = Dict(
        _physical_charge(construction_symmetry, U1Irrep(n)) =>
            edge_coefficient(U1Irrep(n), energy_factor) for n in (-n_max):n_max
    )
    E = energy_tensor(Val(d), O, energy_weights, interaction_weights)

    # convert if necessary
    O = convert_symmetry(symmetry, O)
    E = convert_symmetry(symmetry, E)

    return O, E
end

"""
    classical_villain(
        elt::Type{<:Number}=ComplexF64,
        symmetry::Type{Trivial}=Trivial,
        d::Int=2;
        l_max::Int=1,
        beta::Float64=0.8,
    )

Local tensor corresponding to the partition function of the `d`-dimensional classical
Villain model.
"""
function classical_villain end
function classical_villain(symmetry::Type{<:Sector}; kwargs...)
    return classical_villain(ComplexF64, symmetry, 2; kwargs...)
end
function classical_villain(
    elt::Type{<:Number}=ComplexF64,
    symmetry::Type{<:Sector}=U1Irrep,
    d::Int=2;
    beta::Float64=0.8,
    n_max::Int=1,
    convention::Int=1,
)
    construction_symmetry = symmetry == Trivial ? U1Irrep : symmetry

    # get interaction weights; they're just Bessel functions in this case
    interaction_weights = Dict(
        _physical_charge(construction_symmetry, U1Irrep(n)) =>
            exp(-(n.charge)^2 / (2 * beta)) for n in (-n_max):n_max
    )

    # use these to build the partition function tensor
    O = ON_tensor(elt, construction_symmetry, d, interaction_weights; convention)

    # TODO: also implement the local energy tensor
    @warn "Energy tensor not implemented for Villain model"
    E = zero(O)

    # convert if necessary
    O = convert_symmetry(symmetry, O)
    E = convert_symmetry(symmetry, E)

    return O, E
end
