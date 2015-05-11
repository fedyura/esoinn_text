#ifndef __ADAPT_LEARN_RATE_ABSTRACT_ADAPT_LEARN_RATE__
#define __ADAPT_LEARN_RATE_ABSTRACT_ADAPT_LEARN_RATE__

#include <cstdint>

namespace alr //alr => adapt_learn_rate
{
    //This class returns the learning rate for updating weights
    class AbstractAdaptLearnRate
    {
    public:
        //distance parameter is the distance between the neuron winner and updated neuron
        virtual double getLearnRate(double distance) const = 0;
        
        AbstractAdaptLearnRate(uint32_t iterNumber)
            : m_IterNumber(iterNumber)
        { } 
    protected:
        uint32_t m_IterNumber; //the number of iteration
    };
} //alr

#endif //__ADAPT_LEARN_RATE_ABSTRACT_ADAPT_LEARN_RATE__

