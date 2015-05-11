#include <prep_data/read_dictionary.hpp>

int main (int argc, char* argv[])
{
    const std::string dir_name = "/home/yura/text_anal/prep_words_p2";
    const std::string data_filename = "end_file_p2";
    const std::string dict_filename = "/home/yura/text_anal/word2vec/trunk/word_clusters_end_p2";
    
    std::unordered_map<std::string, std::vector<double>> words_dict;
    if (pd::readWordsVec(dict_filename, words_dict))
        std::cout << "Dictionary is prepared successfully" << std::endl;
    
    std::unordered_map<std::string, std::unordered_map<std::string, double>> distances;
    pd::buildMetricsDict(words_dict, distances);
    std::cout << "Metrics dict is successfully built" << std::endl;
    
    std::unordered_map<std::string, std::vector<std::string>> data;
    if (pd::readPhrases(dir_name, data))
        std::cout << "Phrases is successfully read" << std::endl;
    
    if (pd::writeMetricsFile(data_filename, distances, data))
        std::cout << "End file is prepared" << std::endl;    
        
    
            
    
    //const std::string dir_name = "/home/yura/text_anal/prep_words";
    /*
    const std::string data_filename = "soinn_input";
    const std::string dict_filename = "/home/yura/text_anal/word2vec/trunk/word_vectors";
    
    std::unordered_map<std::string, std::vector<double>> words_dict;
    if (pd::readWordsVec(dict_filename, words_dict))
        std::cout << "Dictionary is prepared successfully" << std::endl;
    
    std::ofstream of(data_filename);
    for (const auto& s: words_dict)
    {
        for (const double coord: s.second)
            of << coord << ",";
        of << s.first << std::endl;    
    }
    */
    
    //if (pd::prepareDataPhrasesFromWordvec(dir_name, data_filename, words_dict))
    //    std::cout << "The file is prepared successfully!!!!" << std::endl;

    /*
    const std::string dir_name = "/home/yura/text_anal/prep_words";
    const std::string data_filename = "soinn_input";
    const std::string dict_filename = "/home/yura/text_anal/word2vec/trunk/word2vec_dict";
    
    std::unordered_map<std::string, std::vector<double>> words_dict;
    if (pd::readWordsVec(dict_filename, words_dict))
        std::cout << "Dictionary is prepared successfully" << std::endl;
    
    if (pd::prepareDataPhrasesFromWordvec(dir_name, data_filename, words_dict))
        std::cout << "The file is prepared successfully!!!!" << std::endl;
    */    
    
    
    
    /*
    const std::string dir_name = "/home/yura/text_anal/prep_words_p2";
    const std::string data_filename = "/home/yura/text_anal/word2vec_train_p2";
    const std::string dict_filename = "/home/yura/text_anal/syn_dict";
    
    std::unordered_map<std::string, std::vector<std::string>> syn_dict;
    if (pd::readSynonymDict(dict_filename, syn_dict))
        std::cout << "Synonym dictionary is prepared successfully" << std::endl;
    
    if (pd::prepareWord2vecInput(dir_name, data_filename, syn_dict))
        std::cout << "The file is prepared successfully!!!!" << std::endl;
        */
        
        
    

    return 0;    
}
