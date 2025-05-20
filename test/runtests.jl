using ONModels
using Test

@testset "ONModels.jl" begin
    @testset "2D models" begin
        @time include("2d/test_xy.jl")
        @time include("2d/test_heis.jl")
        @time include("2d/test_rp2.jl")
    end
    @testset "3D models" begin
        include("3d/pepo_utils.jl")
        @time include("3d/test_xy.jl")
    end
end
