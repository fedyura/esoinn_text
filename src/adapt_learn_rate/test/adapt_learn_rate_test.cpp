#include <boost/test/unit_test.hpp>
#include <adapt_learn_rate/AdaptLearnRateSoinn.hpp>

using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(AdaptLearnRate)

BOOST_AUTO_TEST_CASE(test_SoinnSchema)
{
    alr::AdaptLearnRateSoinn alrs1(5, 4);
    BOOST_CHECK_EQUAL(alrs1.getLearnRate(0), 0.25);

    alr::AdaptLearnRateSoinn alrs2(5, 5);
    BOOST_CHECK_EQUAL(alrs2.getLearnRate(0.1), 0.002);
}

BOOST_AUTO_TEST_SUITE_END()
