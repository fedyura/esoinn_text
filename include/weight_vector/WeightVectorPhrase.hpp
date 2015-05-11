#ifndef __WEIGHT_VECTOR_WEIGHT_VECTOR_PHRASE__
#define __WEIGHT_VECTOR_WEIGHT_VECTOR_PHRASE__

#include <adapt_learn_rate/AbstractAdaptLearnRate.hpp>
#include <container/StaticArray.hpp>
#include <weight_vector/AbstractWeightVector.hpp>
#include <unordered_map>

namespace wv //wv => weight_vector
{
    class WeightVectorPhrase: public AbstractWeightVector
    {
    public:
        uint32_t getNumDimensions() const
        {
            return m_Coords.size();
        }

        std::string getConcreteCoord(uint32_t number) const
        {
            return m_Coords[number]; //m_Coords.at(number); 
        }
        
        //Calculate distance between this weight vector and Point p
        virtual double calcDistance(const Point* p) const;
       
        //Update weight vector in the one training iteration. 
        virtual void updateWeightVector(const Point* p, const alr::AbstractAdaptLearnRate* alr, double distance);

        virtual void eraseOffset();
        virtual double getOffsetValue() const;

        explicit WeightVectorPhrase(const cont::StaticArray<std::string>& coords, const std::unordered_map<std::string, std::unordered_map<std::string, double>>& distances)
            : m_Coords(coords)
            , m_Distances(distances)
        //    , m_Offset(coords)
        {
        //    for (uint32_t i = 0; i < m_Offset.size(); i++)
        //        m_Offset[i] = 0;
        }
   
    private:
        cont::StaticArray<std::string> m_Coords;
        const std::unordered_map<std::string, std::unordered_map<std::string, double>>& m_Distances;
        //cont::StaticArray<double> m_Offset; //coord offset during the one iteration
    };
} //wv

#endif //__WEIGHT_VECTOR_WEIGHT_VECTOR_COSINE__
