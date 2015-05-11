ADD_EXECUTABLE (PrepWordsData prep_data_words.cpp)
TARGET_LINK_LIBRARIES(PrepWordsData ${Boost_LIBRARIES} weight_vector)
