
add_library(wfqueue STATIC
	${CMAKE_CURRENT_LIST_DIR}/wfqueue/align.h
	${CMAKE_CURRENT_LIST_DIR}/wfqueue/primitives.h
	${CMAKE_CURRENT_LIST_DIR}/wfqueue/wfqueue.h
	${CMAKE_CURRENT_LIST_DIR}/wfqueue/wfqueue.c
)

target_compile_definitions(wfqueue PRIVATE WFQUEUE)

set_target_properties(wfqueue PROPERTIES
	C_STANDARD 11
	C_STANDARD_REQUIRED ON
	C_EXTENSIONS OFF
	FOLDER "wfqueue"
)


set(YangMellorCrummeyQueue_short_name YMCQ)
set(YangMellorCrummeyQueue_ref_short_name YMCQ_ref)

set(YangMellorCrummeyQueue_sources ${CMAKE_CURRENT_LIST_DIR}/wait_free_queue.art)
set(YangMellorCrummeyQueue_ref_sources ${CMAKE_CURRENT_LIST_DIR}/wfqueue.art)

set(YangMellorCrummeyQueue_constructor "")
set(YangMellorCrummeyQueue_constructor_u32 "createYangMellorCrummeyQueue")
set(YangMellorCrummeyQueue_ref_constructor "")
set(YangMellorCrummeyQueue_ref_constructor_u32 "createYangMellorCrummeyRefQueue")


set(YangMellorCrummeyQueue_configure_target YangMellorCrummeyQueue_configure)
set(YangMellorCrummeyQueue_ref_configure_target YangMellorCrummeyQueue_ref_configure)

function (YangMellorCrummeyQueue_ref_configure target)
	target_link_libraries(${target} PRIVATE wfqueue)
endfunction()

function (YangMellorCrummeyQueue_configure target)
	YangMellorCrummeyQueue_ref_configure(${target})
endfunction()
