add_custom_target(samples)

function(add_sample)
    cmake_parse_arguments(ARG "" "" "SRCS" ${ARGN})

    set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

    get_filename_component(SAMPLES_DIR ../ ABSOLUTE)
    get_filename_component(SAMPLE_DIR_NAME ${TARGET_DIR} NAME)

    set(CUDA_INSTALL_DIR /usr/local/cuda)


    # add underscores (snake) to camelCase cases
    string(REGEX REPLACE "(.)([A-Z][a-z]+)" "\\1_\\2" SAMPLE_NAME_SNAKE_MIXED ${SAMPLE_DIR_NAME})
    string(REGEX REPLACE "([a-z0-9])([A-Z])" "\\1_\\2" SAMPLE_NAME_SNAKE_MIXED ${SAMPLE_NAME_SNAKE_MIXED})
    string(TOLOWER ${SAMPLE_NAME_SNAKE_MIXED} SAMPLE_NAME_SNAKE)

    # WAR bug http://nvbugs/2518092
    # fix a few sample names
    string(REGEX REPLACE "movie_lens" "movielens" SAMPLE_NAME_FIXED ${SAMPLE_NAME_SNAKE})
    string(REGEX REPLACE "google_net" "googlenet" SAMPLE_NAME_FIXED ${SAMPLE_NAME_FIXED})
    string(REGEX REPLACE "([a-zA-Z0-0])api" "\\1_api" SAMPLE_NAME_FIXED ${SAMPLE_NAME_FIXED})
    string(REGEX REPLACE "_rcnn" "RCNN" SAMPLE_NAME_FIXED ${SAMPLE_NAME_FIXED})

    set(SAMPLE_NAME ${SAMPLE_NAME_FIXED})# CACHE STRING "binary name of the sample")

    set(TARGET_NAME ${SAMPLE_NAME})

    add_executable(${TARGET_NAME}
        ${ARG_SRCS}
        ${SAMPLES_DIR}/common/logger.cpp
        )


    set(ONNX_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include CACHE STRING "Onnx include directory")

    target_include_directories(${TARGET_NAME}
        PUBLIC ${PROJECT_SOURCE_DIR}/include
        PUBLIC ${ONNX_INCLUDE_DIR}
        PUBLIC ${CUDA_INSTALL_DIR}/include
        PRIVATE ${SAMPLES_DIR}/common
        PRIVATE ${TARGET_DIR}
        )

    target_link_libraries(${TARGET_NAME}
        ${CUDART_LIB}
        ${CUBLAS_LIB}
        ${CUDNN_LIB}
        ${NVINFER_LIB}
        ${NVINFER_PLUGIN_LIB}
        ${NVPARSERS_LIB}
        ${NVONNXPARSER_LIB}
        ${RT_LIB}
        ${CMAKE_DL_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
        )
    set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

    set(TRT_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE PATH "sample binary dir")
    set_target_properties(${TARGET_NAME}
        PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${TRT_BIN_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${TRT_BIN_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${TRT_BIN_DIR}
        )

    add_dependencies(samples ${SAMPLE_NAME})
endfunction()

set(SAMPLES_LIST
    sampleCharRNN
    sampleFasterRCNN
    sampleGoogleNet
    sampleINT8
    sampleINT8API
    sampleMLP
    sampleMNIST
    sampleMNISTAPI
    sampleMovieLens
    sampleMovieLensMPS
    sampleOnnxMNIST
    samplePlugin
    sampleReformatFreeIO
    sampleSSD
    sampleUffMNIST
    sampleUffSSD
    trtexec)

foreach(SAMPLE_ITER ${SAMPLES_LIST})
    add_subdirectory(${SAMPLE_ITER})
endforeach()
