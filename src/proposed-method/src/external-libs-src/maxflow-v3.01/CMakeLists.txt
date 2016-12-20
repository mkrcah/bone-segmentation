include_directories (${GRAPHCUTS_SOURCE_DIR}/maxflow-v3.01)

add_library (Graph graph.cpp)
add_library (MaxFlow maxflow.cpp)

