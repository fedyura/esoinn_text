#ifndef __EXAMPLES_READ_DATA_HPP__
#define __EXAMPLES_READ_DATA_HPP__

#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <neuron/AbstractNeuron.hpp>
#include <sstream>
#include <vector>
#include <weight_vector/WeightVectorPhrase.hpp>

namespace fs = boost::filesystem;

namespace ex
{
    std::vector<std::string> split(const std::string& s, char delim)
    {
        std::vector<std::string> elems;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim)) 
            elems.push_back(item);
        
        return elems;
    }

    bool readDataSet(const std::string& dir_name, std::vector<std::shared_ptr<wv::Point>>& points, std::vector<std::string>& answers, const std::unordered_map<std::string, std::unordered_map<std::string, double>>& distances)
    {
        fs::path targetDir(dir_name); 
        fs::directory_iterator it(targetDir), eod;

        BOOST_FOREACH(const fs::path &p, std::make_pair(it, eod))   
        { 
            if(is_regular_file(p))
            {
                if (p.extension() == "swp")
                    continue;
                std::ifstream fi(p.string());
                if (not fi)
                {
                    std::cerr << "File " << p.string() << "doesn't exist" << std::endl;
                    return false;
                }

                std::string line;
                std::vector<std::string> concr_phrase;
                while (std::getline(fi, line) and !line.empty())
                {
                    concr_phrase.push_back(line);
                }
                
                cont::StaticArray<std::string> arr(concr_phrase.size());
                for (uint32_t i = 0; i < concr_phrase.size(); i++)
                    arr[i] = concr_phrase[i];
                    
                answers.push_back(p.filename().string());
                std::shared_ptr<wv::WeightVectorPhrase> swv(new wv::WeightVectorPhrase(arr, distances));
                points.push_back(swv);
            } 
        }
        return true;
    }
}
#endif //__EXAMPLES_READ_DATA_HPP__
