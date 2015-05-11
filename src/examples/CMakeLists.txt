ADD_EXECUTABLE (ESoinnIris esoinn_iris.cpp)
TARGET_LINK_LIBRARIES(ESoinnIris esoinn ${Boost_LIBRARIES} weight_vector)
