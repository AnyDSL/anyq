
add_library(moodycamel STATIC
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/concurrentqueue.h
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/c_api/concurrentqueue.h
	${CMAKE_CURRENT_LIST_DIR}/moodycamel/c_api/concurrentqueue.cpp
)

target_compile_definitions(moodycamel PRIVATE MOODYCAMEL_STATIC)

set(MoodyCamelQueue_short_name MOQ)

set(MoodyCamelQueue_sources ${CMAKE_CURRENT_LIST_DIR}/moodycamel.art)

set(MoodyCamelQueue_constructor "createMoodyCamelQueueGeneric")
set(MoodyCamelQueue_constructor_u32 "createMoodyCamelIndexQueue")

set(MoodyCamelQueue_configure_target MoodyCamelQueue_configure)

function (MoodyCamelQueue_configure target)
	target_link_libraries(${target} PRIVATE moodycamel)
endfunction()
