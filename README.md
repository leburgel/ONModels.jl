# ONModels.jl

Tensor-network implementation of the partition function of some classical $O(N)$ models,

$$
\mathcal{Z}(\\beta) = \sum_{\{s\}} \\exp(-\beta H(s)) \\text{ with } H(s) = -\\sum_{\langle i, j \\rangle} \\left ( \\vec{s}_i \\cdot \\vec{s}_j \\right )^p
$$

where $\vec{s}_i$ denotes an $N$-component classical spin of unit length at site $i$
of the $d$-dimensional hypercubic lattice.

In $d=2$ dimensions, we provide implementations of the constituent partition function tensor
for the classical $XY$ ($N=2, p=1$), Heisenberg ($N=3, p=1$) and $RP^2$ ($N=3, p=1$) models.
These can be combined with [MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl) and
[TNRKit.jl](https://github.com/VictorVanthilt/TNRKit.jl) to, for example, reproduce some of
the results of:

- [Phys. Rev. E 100, 062136 (2019)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.100.062136)
- [Phys. Rev. B 104, 165132 (2021)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.165132)
- [SciPost Phys. 11, 098 (2021)](https://scipost.org/10.21468/SciPostPhys.11.5.098)
- [Phys. Rev. E 106, 014104 (2022)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.106.014104)
- [Phys. Rev. E 107, 014117 (2023)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.107.014117)

For $d=3$, the constituent tensors can be combined with
[PEPSKit.jl](https://github.com/QuantumKitHub/PEPSKit.jl) to contract the corresponding
infinite cubic partition function using a method similar to that or
[Phys. Rev. E 98, 042145 (2018)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.98.042145).
