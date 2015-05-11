#include <boost/format.hpp>
#include <examples/read_data.hpp>
#include <cmath>
#include <logger/logger.hpp>
#include <neural_network/ESoinn.hpp>
#include <prep_data/read_dictionary.hpp>
#include <stdlib.h>
#include <time.h>

namespace 
{
    logger::ConcreteLogger* log_netw = logger::Logger::getLog("ESoinn", logger::LogLevels::INFO);
    
    std::unordered_map<uint32_t, uint32_t> neuron_clusters; //neuron => cluster
    const uint32_t NumDimensionsIrisDataset = 4;
    //const uint32_t NumDimensionsIrisDataset = 100;

    const std::string dir_name = "/home/yura/text_anal/prep_words3";
    const std::string data_filename = "end_file_p2";
    const std::string dict_filename = "/home/yura/text_anal/word2vec/trunk/word_clusters_end_p2";

}

int main (int argc, char* argv[])
{
    srand (time(NULL));
    //FormMCLInputFile("mcl_input");
    
    std::vector<std::shared_ptr<wv::Point>> points;
    std::vector<std::string> answers;

    log_netw->info("Hello world");
    
    //if (!ex::readDataSet("iris_dataset", NumDimensionsIrisDataset, points, answers))
    std::unordered_map<std::string, std::vector<double>> words_dict;
    if (pd::readWordsVec(dict_filename, words_dict))
        std::cout << "Dictionary is prepared successfully" << std::endl;
    
    std::unordered_map<std::string, std::unordered_map<std::string, double>> distances;
    pd::buildMetricsDict(words_dict, distances);
    std::cout << "Metrics dict is successfully built" << std::endl;

    if (!ex::readDataSet(dir_name, points, answers, distances))
    {
        std::cerr << "readIrisDataSet function works incorrect" << std::endl;
    }
    
    std::string output_filename = "points";
    //25, 50, 1, 0.1
    nn::ESoinn ns(NumDimensionsIrisDataset, 40 /*age_max*/, 2500 /*lambda*/, 0.1 /*C1*/, 0.01 /*C2*/,
                  nn::NetworkStopCriterion::LOCAL_ERROR, neuron::NeuronType::PHRASE, distances); 
        
    std::vector<std::vector<uint32_t>> conn_comp;
    ns.trainNetwork(points, conn_comp, answers, 2);  
    ns.dumpNetwork(); 
    
    for (uint32_t i = 0; i < conn_comp.size(); i++)
        for (uint32_t j = 0; j < conn_comp[i].size(); j++)
            neuron_clusters.emplace(conn_comp[i][j], i);
        
    //define clusters
    uint32_t i = 0;
    
    std::unordered_map<std::string, std::string> phrases;
    pd::readPhrases("/home/yura/text_anal/quotes", phrases);
    
    std::multimap<uint32_t, std::string> classes;
    for (const auto p: points)
    {
        //log_netw->info((boost::format("%d %s") % ns.findPointCluster(p.get(), neuron_clusters) % answers[i]).str());
        classes.emplace(ns.findPointCluster(p.get(), neuron_clusters), phrases[answers[i]]);
        i++;
        //if (i % 50 == 0) log_netw->info("\n", true);
    }
    
    ns.exportNetworkGDF("network.gdf", neuron_clusters);
                
    for (const auto& s: classes)
        log_netw->info((boost::format("%d %s") % s.first % s.second).str());
    log_netw->info("\n", true);
    log_netw->info("The program is succesfully ended");
    
    return 0;
}
