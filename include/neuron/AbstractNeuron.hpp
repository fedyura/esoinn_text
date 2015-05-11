#ifndef __NEURON_ABSTRACT_NEURON__
#define __NEURON_ABSTRACT_NEURON__

#include <weight_vector/AbstractWeightVector.hpp>

namespace neuron
{
    enum NeuronType
    {
        EUCLIDEAN = 1,
        COSINE    = 2,
        PHRASE    = 3
    };
    
    class AbstractNeuron
    {
    public:
        AbstractNeuron(wv::AbstractWeightVector *wv)
            : m_wv(wv)
            , m_CurPointDist(-1)
        { }
        
        wv::AbstractWeightVector* getWv() const
        {
            return m_wv;
        }

        //throw std::runtime_error or std::bad_typeid in the case of error
        double setCurPointDist(const wv::Point* p)
        {
            m_CurPointDist = m_wv->calcDistance(p);
            return m_CurPointDist;
        }

        double curPointDist()
        {
            return m_CurPointDist;
        }

        void setZeroPointer()
        {
            m_wv = NULL;
        }

    protected:
        wv::AbstractWeightVector* m_wv = NULL;
        double m_CurPointDist; //distance to the current point (or to neuron winner)        
    };
} //neuron

#endif //__NEURON_ABSTRACT_NEURON__
