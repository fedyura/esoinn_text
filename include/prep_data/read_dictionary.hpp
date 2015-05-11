#ifndef __PREP_DATA_READ_DICTIONARY_HPP__
#define __PREP_DATA_READ_DICTIONARY_HPP__

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <cmath>
#include <fstream>
#include <functional>
#include <examples/read_data.hpp>
#include <map>
#include <string>
#include <unordered_map>

namespace fs = boost::filesystem;

namespace pd
{
    bool readDictionary(const std::string& dict_filename, std::unordered_map<std::string, uint32_t>& dict)
    {
        std::ifstream fi(dict_filename);
        if (not fi)
        {
            std::cerr << "File " << dict_filename << " doesn't exist" << std::endl;
            return false;
        }

        std::string line;
        uint32_t linesread = 1;
        while (std::getline(fi, line) and !line.empty())
        {
            dict.emplace(line, linesread);
            linesread++;
        }
        return true;
    }

    bool prepareData(const std::string& dir_name, const std::string& data_filename, const std::string& dict_filename, const uint32_t num_dimensions)
    {
        std::unordered_map<std::string, uint32_t> dict;
        if (!readDictionary(dict_filename, dict))
            return false;
        
        fs::path targetDir(dir_name); 
        fs::directory_iterator it(targetDir), eod;

        std::ofstream of(data_filename);
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
                std::map<uint32_t, uint32_t> result;
                std::hash<uint32_t> uint32_hash;
                while (std::getline(fi, line) and !line.empty())
                {
                    const std::unordered_map<std::string, uint32_t>::const_iterator it = dict.find(line);
                    if (it == dict.end())
                        continue;
                    
                    uint32_t key = uint32_hash(it->second) % num_dimensions;
                    result[key]++;
                }
                
                for (uint32_t i = 0; i < num_dimensions; i++)
                {
                    const std::map<uint32_t, uint32_t>::const_iterator it = result.find(i);
                    if (it == result.end())
                        of << "0,";
                    else
                        of << std::to_string(it->second) << ",";    
                }
                of << p.filename() << std::endl;
            } 
        }
        return true;
    }

    bool readWordsVec(const std::string& dict_filename, std::unordered_map<std::string, std::vector<double>>& words_dict)
    {
        std::ifstream fi(dict_filename);
        if (not fi)
        {
            std::cerr << "File " << dict_filename << " doesn't exist" << std::endl;
            return false;
        }

        std::string line;
        
        //read first line and get number of features
        std::getline(fi, line);
        std::vector<std::string> items;
        items = ex::split(line, ' ');
        if (items.size() != 2)
            std::cerr << "File " << dict_filename << " has incorrect format (wrong first line)" << std::endl;
        
        uint32_t num_features = 0;
        try
        {
            num_features = std::stoull(items[1]);
        }
        catch (std::invalid_argument& exc)
        {
            std::cerr << "Incorrect first line of file " << dict_filename << std::endl;
        }
        
        while (std::getline(fi, line) and !line.empty())
        {
            std::vector<std::string> coords = ex::split(line, ' ');
            if (coords.size() != num_features + 1)
            {
                std::cerr << "Incorrect line " << line << std::endl; 
                continue;
            }
            
            std::string word = coords[0];
            coords.erase(coords.begin(), coords.begin() + 1);
            std::vector<double> items; 
            for (const std::string& s: coords)
            {
                std::istringstream is(s);
                double x = 0;
                if (!(is >> x))
                {
                    std::cerr << "Line " << line << " has incorrect numbers" << std::endl;
                    break;
                }
                items.push_back(x);
            }                     
            words_dict.emplace(word, items);
        }        
        return true;
    }
    
    bool prepareDataPhrasesFromWordvec(const std::string& dir_name, const std::string& data_filename, const std::unordered_map<std::string, std::vector<double>>& words_dict)
    {        
        fs::path targetDir(dir_name); 
        fs::directory_iterator it(targetDir), eod;

        std::ofstream of(data_filename);
        BOOST_FOREACH(const fs::path &p, std::make_pair(it, eod))   
        { 
            if(is_regular_file(p))
            {
                std::cout << p.string() << std::endl;
                if (p.extension() == "swp")
                    continue;
                std::ifstream fi(p.string());
                if (not fi)
                {
                    std::cerr << "File " << p.string() << "doesn't exist" << std::endl;
                    return false;
                }

                std::string line;
                std::vector<double> result;
                while (std::getline(fi, line) and !line.empty())
                {
                    const std::unordered_map<std::string, std::vector<double>>::const_iterator it = words_dict.find(line);
                    if (it == words_dict.end())
                    {
                        std::cerr << "word " << line << " in file " << p.string() << " isn't found in dictionary" << std::endl;
                        //return false;
                        continue;
                    }
                    
                    if (result.size() == 0)
                        result = it->second;
                    else
                        std::transform (result.begin(), result.end(), it->second.begin(), result.begin(), std::plus<double>());                            
                }
                
                for (const double elem: result)
                    of << elem << ",";
                of << p.filename() << std::endl;
            } 
        }
        return true;
    }

    bool readPhrases(const std::string& dir_name, std::unordered_map<std::string, std::string>& phrases)
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
                std::getline(fi, line);
                std::string cur_filename = p.filename().string();
                cur_filename.replace(0, 6, "words");
                phrases.emplace(cur_filename, line);
            } 
        }
        return true;
    }
    
    bool readSynonymDict(const std::string& dict_filename, std::unordered_map<std::string, std::vector<std::string>>& dict) 
    {
        std::ifstream fi(dict_filename);
        if (not fi)
        {
            std::cerr << "File " << dict_filename << " doesn't exist" << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(fi, line) and !line.empty())
        {
            //boost::algorithm::trim(line);
            line.erase(line.find_last_not_of(" \n\r\t")+1);
            //std::cout << line << std::endl;
            std::vector<std::string> coords = ex::split(line, '|');
            std::string word = coords[0];
            coords.erase(coords.begin(), coords.begin() + 1);
            uint32_t num_words = 0;
            std::vector<std::string> synonims;
            for (const std::string& s: coords)
            {
                if (s.find(' ') != std::string::npos or s.find('(') != std::string::npos or s.find(')') != std::string::npos)
                    continue;
                synonims.push_back(s);    
                num_words++;
                if (num_words == 5)
                    break;
            }
            
            dict.emplace(word, synonims);               
        }
        return true;
    }

    bool prepareWord2vecInput(const std::string& dir_name, const std::string& out_filename, const std::unordered_map<std::string, std::vector<std::string>>& dict)
    {
        fs::path targetDir(dir_name); 
        fs::directory_iterator it(targetDir), eod;

        std::ofstream of(out_filename);
        BOOST_FOREACH(const fs::path &p, std::make_pair(it, eod))   
        { 
            if(is_regular_file(p))
            {
                //std::cout << p.string() << std::endl;
                if (p.extension() == "swp")
                    continue;
                std::ifstream fi(p.string());
                if (not fi)
                {
                    std::cerr << "File " << p.string() << "doesn't exist" << std::endl;
                    return false;
                }

                std::string line;
                while (std::getline(fi, line) and !line.empty())
                {
                    of << line;
                    std::unordered_map<std::string, std::vector<std::string>>::const_iterator it = dict.find(line);
                    if (it == dict.end())
                        continue;
                    
                    for (const auto& s: it->second)
                    {
                        of << " " << s;
                    }
                    of << " ";
                }                
            }
            of << std::endl; 
        }
        return true;
    }
    
    double cosineMetric(const std::vector<double>& word1, const std::vector<double>& word2)
    {
        if (word1.size() != word2.size())
            throw std::runtime_error("Error in cosineMetric function. Two words have different size");
        double inner_product = 0.0, norm1 = 0.0, norm2 = 0.0;
        for (uint32_t i = 0; i < word1.size(); i++)
        {
            inner_product += word1[i] * word2[i];
            norm1 += word1[i] * word1[i];
            norm2 += word2[i] * word2[i];
        }
        return std::abs(inner_product) / sqrt (norm1 * norm2);            
    }
    
    void buildMetricsDict(const std::unordered_map<std::string, std::vector<double>>& dict, std::unordered_map<std::string, std::unordered_map<std::string, double>>& distances)
    {
        for (const auto& s1: dict)
        {
            for (const auto& s2: dict)
            {
                double dist = cosineMetric(s1.second, s2.second);
                if (dist > 0.9)
                    distances[s1.first][s2.first] = dist;
            }
        }
    }

    double diffTwoPhrases(const std::vector<std::string>& phrase1, const std::vector<std::string>& phrase2, std::unordered_map<std::string, std::unordered_map<std::string, double>>& distances)
    {
        double metric = 0.0;
        for (const std::string& s1: phrase1)
        {
            for (const std::string& s2: phrase2)
            {
                metric += distances[s1][s2];
            }
        }
        //std::cout << "Metrics = " << metric << std::endl;
        return metric;
    }

    double metricPhrases(const std::vector<std::string>& phrase1, const std::vector<std::string>& phrase2, std::unordered_map<std::string, std::unordered_map<std::string, double>>& distances)
    {
        //return diffTwoPhrases(phrase1, phrase2, distances) / sqrt(diffTwoPhrases(phrase1, phrase1, distances) * diffTwoPhrases(phrase2, phrase2, distances));
        return diffTwoPhrases(phrase1, phrase2, distances); // (phrase1.size() * phrase2.size());
    }

    bool readPhrases(const std::string& dir_name, std::unordered_map<std::string, std::vector<std::string>>& data)
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
                    //line.erase(line.find_last_not_of(" \n\r\t")+1);
                    concr_phrase.push_back(line);
                }
                data.emplace(p.filename().string(), concr_phrase);
                concr_phrase.clear();
            } 
        }
        return true;
    }

    bool writeMetricsFile(const std::string& filename, std::unordered_map<std::string, std::unordered_map<std::string, double>>& distances, const std::unordered_map<std::string, std::vector<std::string>>& data)
    {
        std::ofstream of(filename);
        for (const auto& s1: data)
        {
            for (const auto& s2: data)
            {
                double dist = metricPhrases(s1.second, s2.second, distances);
                if (dist > 1.5 && dist < 10)
                    of << s1.first << " " << s2.first << " " << dist << std::endl;
            }
        }
        return true;
    }
}
#endif //__PREP_DATA_READ_DICTIONARY_HPP__
