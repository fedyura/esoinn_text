#ifndef __WEIGHT_VECTOR_ABSTRACT_WEIGHT_VECTOR__
#define __WEIGHT_VECTOR_ABSTRACT_WEIGHT_VECTOR__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <cmath>

namespace wv //wv => weight_vector
{
    class AbstractWeightVector;
    
    //WeightVector and Point is a equal things in terms of implementation
    //There is only semantic difference. WeightVector is a parameters of neuron; Point is a input data in specified space
    typedef AbstractWeightVector Point;
     
    class AbstractWeightVector
    {
    public:
        //Calculate distance between this weight vector and Point p
        virtual double calcDistance(const Point* p) const = 0;
        
        //Update weight vector in the one training iteration. Return false if error
        virtual void updateWeightVector(const Point* p, const alr::AbstractAdaptLearnRate* alr, double distance) = 0;

        //Get concrete coord
        virtual std::string getConcreteCoord(uint32_t number) const = 0;
                
        virtual uint32_t getNumDimensions() const = 0;
 
        virtual void eraseOffset() = 0;
        virtual double getOffsetValue() const = 0;

        virtual ~AbstractWeightVector() { };
    };
        
} //wv

#endif //__WEIGHT_VECTOR_ABSTRACT_WEIGHT_VECTOR__


