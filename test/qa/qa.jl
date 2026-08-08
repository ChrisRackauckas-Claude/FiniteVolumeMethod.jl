using SciMLTesting, FiniteVolumeMethod, Test

@test SciMLTesting.public_reexports(FiniteVolumeMethod) == [:solve]

run_qa(
    FiniteVolumeMethod;
    reexports_allow = (:solve,),
)
