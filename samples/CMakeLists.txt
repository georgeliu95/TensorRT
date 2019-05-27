add_custom_target(samples)

add_subdirectory(opensource)

if(NVINTERNAL OR NVPARTNER)
    add_subdirectory(proprietary)
endif()