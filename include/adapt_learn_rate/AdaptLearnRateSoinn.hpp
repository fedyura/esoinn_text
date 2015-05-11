#ifndef __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_SOINN__
#define __ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_SOINN__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <cmath>

namespace alr //alr => adapt_learn_rate
{
    class AdaptLearnRateSoinn: public AbstractAdaptLearnRate
    {
    public:
        virtual double getLearnRate(double distance) const
        {
            return (distance == 0) ? 1.0 / m_NumWinner : 1.0 / (100 * m_NumWinner);
        }
        
        AdaptLearnRateSoinn(uint32_t iterNumber, uint32_t numWinner)
            : AbstractAdaptLearnRate(iterNumber)
            , m_NumWinner(numWinner)
        { } 
    private:
        uint32_t m_NumWinner; //number of input signals for which node was a winner (M_i)
    };
} //alr

#endif //__ADAPT_LEARN_RATE_ADAPT_LEARN_RATE_SOINN__
