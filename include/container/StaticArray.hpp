#ifndef __CONTAINERS_STATIC_ARRAY_HPP__
#define __CONTAINERS_STATIC_ARRAY_HPP__

#include <algorithm>
#include <stdexcept>
#include <stdint.h>

namespace cont //containers
{
    template<typename T>
    class StaticArray
    {
    public:
        uint32_t size() const
        {   
            return m_Size;
        }
        
        //check bounds. Throw out_of_range exception if position is out of range 
        T& at(uint32_t index)
        {
            if (index >= m_Size) 
                throw std::out_of_range("Too large index");
            return m_MasPoint[index];    
        }
        
        const T& at(uint32_t index) const
        {
            if (index >= m_Size)
                throw std::out_of_range("Too large index");
            return m_MasPoint[index];    
        }
        
        //doesn't check bounds
        T& operator[](uint32_t index)
        {
            return m_MasPoint[index];
        }
        const T& operator[](uint32_t index) const
        {
            return m_MasPoint[index];
        }

        T* data()
        {
            return m_MasPoint;
        }
        
        explicit StaticArray(uint32_t size);
        StaticArray(const StaticArray& sa);
        StaticArray& operator=(StaticArray sa);
        StaticArray() { };

        ~StaticArray();

        friend void swap(StaticArray& first, StaticArray& second) //nothrow
        {
            using std::swap;
            swap(first.m_Size, second.m_Size);
            swap(first.m_MasPoint, second.m_MasPoint);
        }

    private:
        T* m_MasPoint = NULL;
        uint32_t m_Size = 0;
    };

    template <typename T>
    StaticArray<T>::StaticArray(uint32_t size)
        : m_Size(size)
    {
        m_MasPoint = new T[m_Size];
    }

    template <typename T>
    StaticArray<T>::~StaticArray()
    {
        if (m_MasPoint != NULL)
            delete[] m_MasPoint;
    }

    template <typename T>
    StaticArray<T>& StaticArray<T>::operator=(StaticArray sa) //copy and swap idiom
    {
        swap(*this, sa);
        return *this;
    }

    template <typename T>
    StaticArray<T>::StaticArray(const StaticArray& sa)
    {
        m_Size = sa.size();
        m_MasPoint = new T[m_Size];
        for (uint32_t i = 0; i < m_Size; i++)
            m_MasPoint[i] = sa[i];
    }

} //cont

#endif //__CONTAINERS_STATIC_ARRAY_HPP__
