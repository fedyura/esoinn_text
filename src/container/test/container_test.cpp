#include <boost/test/unit_test.hpp>
#include <container/StaticArray.hpp>

using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(Container)

BOOST_AUTO_TEST_CASE(test_StaticArrayWork)
{
    uint32_t size = 3;
    cont::StaticArray<double> sa(size);
    sa[0] = 5.1;
    sa[1] = 7.2;
    sa[2] = 9.3;

    BOOST_CHECK_EQUAL(sa.size(), 3);
    
    BOOST_CHECK_EQUAL(sa[0], 5.1);
    BOOST_CHECK_EQUAL(sa.at(2), 9.3);
    BOOST_CHECK_THROW(sa.at(3), std::out_of_range); 
}

BOOST_AUTO_TEST_CASE(test_CopyStaticArray)
{
    uint32_t size = 4;
    cont::StaticArray<uint32_t> arr(size);
    arr[0] = 1;
    arr[1] = 2;
    arr[2] = 3;
    arr[3] = 4;
    
    cont::StaticArray<uint32_t> arr2(1);
    BOOST_CHECK_EQUAL(arr2.size(), 1);
    
    arr2 = arr; 
    BOOST_CHECK(std::equal(arr.data(), arr.data() + arr.size(), arr2.data()));

    cont::StaticArray<uint32_t> arr3(arr);
    BOOST_CHECK(std::equal(arr.data(), arr.data() + arr.size(), arr3.data()));
    
    cont::StaticArray<uint32_t> arr4 = arr;
}

BOOST_AUTO_TEST_SUITE_END()
