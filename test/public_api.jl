using Test
import FiniteVolumeMethod

module ExternalFiniteVolumeClient
    import FiniteVolumeMethod: AbstractFVMProblem, AbstractFVMTemplate, get_dirichlet_nodes,
        get_volume, solve

    struct ClientMesh end
    struct ClientConditions end

    struct ClientProblem <: AbstractFVMProblem
        mesh::ClientMesh
        conditions::ClientConditions
    end

    struct ClientTemplate{P <: AbstractFVMProblem} <: AbstractFVMTemplate
        problem::P
    end

    get_volume(::ClientMesh, node) = 10node
    get_dirichlet_nodes(::ClientConditions) = (2, 4)
    solve(::ClientProblem, value; scale = 1) = scale * value
end

@testset "External abstract interface" begin
    client_problem = ExternalFiniteVolumeClient.ClientProblem(
        ExternalFiniteVolumeClient.ClientMesh(),
        ExternalFiniteVolumeClient.ClientConditions(),
    )
    client_template = ExternalFiniteVolumeClient.ClientTemplate(client_problem)

    @test FiniteVolumeMethod.solve(client_problem, 3; scale = 2) == 6
    @test FiniteVolumeMethod.solve(client_template, 3; scale = 2) == 6
    @test FiniteVolumeMethod.get_volume(client_problem, 3) == 30
    @test FiniteVolumeMethod.get_dirichlet_nodes(client_problem) == (2, 4)
    @test !Base.isexported(FiniteVolumeMethod, :AbstractFVMProblem)
    @test !Base.isexported(FiniteVolumeMethod, :AbstractFVMTemplate)

    @static if VERSION >= v"1.11.0-DEV.469"
        @test Base.ispublic(FiniteVolumeMethod, :AbstractFVMProblem)
        @test Base.ispublic(FiniteVolumeMethod, :AbstractFVMTemplate)
    end
end
