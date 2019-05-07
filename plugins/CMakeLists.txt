add_custom_target(plugin)

set(TARGET_NAME nvinfer_plugin)

set(TARGET_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(PLUGIN_SOURCES
  clipPlugin.cpp
  gridAnchorPlugin.cpp
  InferPlugin.cpp
  nmsPlugin.cpp
  normalizePlugin.cpp
  nvPluginsLegacy.cpp
  lReluPlugin.cpp
  priorBoxPlugin.cpp
  regionPlugin.cpp
  reorgPlugin.cpp
  vocEvaluate.cpp
  batchedNMSPlugin.cpp
  nmsHelper.cpp
  checkMacrosPlugin.cpp
  reducedMathPlugin.cpp
  allClassNMS.cu
  bboxDeltas2Proposals.cu
  clip.cu
  common.cu
  decodeBBoxes.cu
  detectionForward.cu
  extractFgScores.cu
  gatherNMSOutputs.cu
  gatherTopDetections.cu
  generateAnchors.cu
  nmsLayer.cu
  normalizeLayer.cu
  NvPluginFasterRCNN.cu
  NvPluginSSD.cu
  permuteData.cu
  permuteLayer.cu
  lReLU.cu
  priorBoxLayer.cu
  proposalsForward.cu
  regionForward.cu
  reorgForward.cu
  roiPooling.cu
  rproiInferenceFused.cu
  sortScoresPerClass.cu
  sortScoresPerImage.cu
  batchedNMSInference.cu
)

set(CUDA_INSTALL_DIR /usr/local/cuda)

add_library(${TARGET_NAME} SHARED
    ${PLUGIN_SOURCES}
)

target_include_directories(${TARGET_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${CUB_ROOT_DIR} 
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PRIVATE ${TARGET_DIR}
)

set_target_properties(${TARGET_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    ARCHIVE_OUTPUT_DIRECTORY ${TRT_LIB_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${TRT_LIB_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${TRT_LIB_DIR}
)

set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX ${TRT_DEBUG_POSTFIX})

set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${TRT_VERSION} SOVERSION ${TRT_SOVERSION} )

set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_STANDARD 11)

add_dependencies(plugin ${TARGET_NAME})
