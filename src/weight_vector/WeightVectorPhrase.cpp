#include <exception>
#include <cstring>
#include <typeinfo>
#include <weight_vector/WeightVectorPhrase.hpp>

namespace wv
{
    double WeightVectorPhrase::calcDistance(const Point* p) const
    {
        if (strcmp(typeid(*p).name(), typeid(*this).name()) != 0)
            throw std::bad_typeid();
        
        double metric = 0;
        for (uint32_t i = 0; i < getNumDimensions(); i++)
        {
            for (uint32_t j = 0; j < p->getNumDimensions(); j++)
            {
                std::unordered_map<std::string, std::unordered_map<std::string, double>>::const_iterator it_ext = m_Distances.find(getConcreteCoord(i));
                if (it_ext != m_Distances.end())
                {
                    std::unordered_map<std::string, double>::const_iterator it_int = it_ext->second.find(p->getConcreteCoord(j));
                    if (it_int != it_ext->second.end())
                        metric += it_int->second;
                }
            }
        }
        
        return 5.0 / (metric + 1.0); 
    }

    //distance - distance between neuron winner and point p
    void WeightVectorPhrase::updateWeightVector(const Point* p, const alr::AbstractAdaptLearnRate* alr, double distance)
    {
        if (strcmp(typeid(*p).name(), typeid(*this).name()) != 0)
            throw std::bad_typeid();
        
        if (getNumDimensions() < 2 || p->getNumDimensions() < 2)
            return;
        
        //uint32_t i = std::rand() % getNumDimensions();
        //uint32_t j = std::rand() % p->getNumDimensions();
        //m_Coords[i] = p->getConcreteCoord(j);

        //uint32_t i = 0, j = 0, max_dist = 0;
        uint32_t max_dist = 0;
        uint32_t i_coord = 0, j_coord = 0;
        for (uint32_t i = 0; i < getNumDimensions(); i++)
        {
            for (uint32_t j = 0; j < p->getNumDimensions(); j++)
            {
                std::unordered_map<std::string, std::unordered_map<std::string, double>>::const_iterator it_ext = m_Distances.find(getConcreteCoord(i));
                if (it_ext != m_Distances.end())
                {
                    std::unordered_map<std::string, double>::const_iterator it_int = it_ext->second.find(p->getConcreteCoord(j));
                    if (it_int != it_ext->second.end())
                        if (max_dist < it_int->second)
                        {
                            max_dist = it_int->second;
                            i_coord = i;
                            j_coord = j;
                        }
                }
            }
        }
        //if (max_dist > 0.7 && max_dist < 1)
            m_Coords[i_coord] = p->getConcreteCoord(j_coord);
    }

    void WeightVectorPhrase::eraseOffset()
    {
        /*for (uint32_t i = 0; i < m_Offset.size(); i++)
            m_Offset[i] = 0;
            */
    }
    
    double WeightVectorPhrase::getOffsetValue() const 
    {
        /*
        double value = 0;
        for (uint32_t i = 0; i < m_Offset.size(); i++)
            value += m_Offset[i] * m_Offset[i];
        return value;    
        */
        return 0.0;
    }
} //wv
