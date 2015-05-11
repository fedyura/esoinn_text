#ifndef __NEURON_ESOINN_NEURON_HPP__
#define __NEURON_ESOINN_NEURON_HPP__

#include <assert.h>
#include <neuron/AbstractNeuron.hpp>
#include <string>
#include <unordered_map>

namespace neuron
{
    class ESoinnNeuron: public AbstractNeuron
    {
      public:
        ESoinnNeuron(wv::AbstractWeightVector* wv, std::string label)
          : AbstractNeuron(wv)
          , m_IsDeleted(false)
          , m_Label(label)
          , m_CurClass(0) 
          , m_NeighbourMaxDensity(0) 
          , m_LocalSignals(1)
          , m_Density(0)
        { }
            
        ESoinnNeuron()
          : AbstractNeuron(NULL)
          , m_IsDeleted(false)
          , m_Label("0")
          , m_CurClass(0)
          , m_NeighbourMaxDensity(0) 
          , m_LocalSignals(1)
          , m_Density(0)
        { }

        //increment age of edges for this neuron and return vector of neighbours
        std::vector<uint32_t> incrementEdgesAge()
        {
            std::vector<uint32_t> neighbours;
            for (auto& s: m_Neighbours)
            {
                s.second++;
                neighbours.push_back(s.first);
            }
            return neighbours;
        }
        
        std::vector<uint32_t> getNeighbours() const
        {
            std::vector<uint32_t> neighbours;
            for (auto& s: m_Neighbours)
                neighbours.push_back(s.first);
            return neighbours;
        }

        //increment age of concrete edge
        //throw std::out_of_range if this edge isn't found
        void incrementConcreteEdgeAge(uint32_t number)
        {
            m_Neighbours.at(number)++;
        }

        void updateEdge(int number)
        {
            //null age of existing edge or create new edge with null age
            m_Neighbours[number] = 0;
        }

        void deleteOldEdges(uint32_t age_max)
        {
            std::unordered_map<uint32_t, uint32_t>::iterator it = m_Neighbours.begin();
            while (it != m_Neighbours.end())
            {
                if (it->second > age_max)
                    it = m_Neighbours.erase(it);
                else 
                    it++;    
            }
        }

        void replaceNeighbour(uint32_t number_old, uint32_t number_new, bool save_old_edge_age = false)
        {
            std::unordered_map<uint32_t, uint32_t>::iterator it = m_Neighbours.find(number_old);
            if (it == m_Neighbours.end()) 
                throw std::runtime_error("Error in replaceNeighbour function. Old edge doesn't exist");
            
            if (!save_old_edge_age)
                m_Neighbours.emplace(number_new, 0);  //it->second (if set old value)
            else
                m_Neighbours.emplace(number_new, it->second);  //it->second (if set old value)
            m_Neighbours.erase(it);
        }

        void deleteConcreteNeighbour(uint32_t number)
        {
            m_Neighbours.erase(number);
        }
        
        void incrementLocalSignals()
        {
            m_LocalSignals++;
        }

        void evalAndSetDensity(double distance)
        {
            m_Density = (m_Density * (m_LocalSignals - 1) + 1.0 / ((1 + distance) * (1 + distance))) / m_LocalSignals;
        }

        double density() const
        {
            return m_Density;
        }

        //for tests
        void setDensity(double density)
        {
            m_Density = density;
        }

        void setNeighbourMaxDensity(uint32_t neighbour)
        {
            m_NeighbourMaxDensity = neighbour;
        }

        uint32_t neighbourMaxDensity()
        {
            return m_NeighbourMaxDensity;
        }

        void setCurClass(uint32_t cur_class)
        {
            m_CurClass = cur_class;
        }

        void setDeleted()
        {
            m_IsDeleted = true;
        }

        bool is_deleted() const
        {
            return m_IsDeleted;
        }

        uint32_t curClass() const
        {
            return m_CurClass;
        }

        double localSignals() const
        {
            return m_LocalSignals;
        }

        uint32_t getNumNeighbours() const
        {
            return m_Neighbours.size();
        }

        std::string label() const
        {
            return m_Label;
        }

        std::unordered_map<uint32_t, uint32_t> getNeighboursAge() const
        {
            return m_Neighbours;
        }

      private:        
        bool m_IsDeleted;
        std::string m_Label;
        uint32_t m_CurClass;
        uint32_t m_NeighbourMaxDensity;                   
        double m_LocalSignals;   
        double m_Density;     
        
        std::unordered_map<uint32_t, uint32_t> m_Neighbours; //Neigbour number => edge age
    };
} //neuron

#endif //__NEURON_ESOINN_NEURON_HPP__
