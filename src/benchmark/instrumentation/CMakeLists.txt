

if (cpu IN_LIST AnyQ_PLATFORMS OR fiberless IN_LIST AnyQ_PLATFORMS)
	add_subdirectory(cpu)
	add_library(instrumentation_fiberless ALIAS instrumentation_cpu)
endif()

if (cuda IN_LIST AnyQ_PLATFORMS OR nvvm IN_LIST AnyQ_PLATFORMS)
	add_subdirectory(cuda)
endif()

if (nvvm IN_LIST AnyQ_PLATFORMS)
	add_library(instrumentation_nvvm ALIAS instrumentation_cuda)
endif()

if (amdgpu IN_LIST AnyQ_PLATFORMS)
	add_subdirectory(amdgpu)
endif()
