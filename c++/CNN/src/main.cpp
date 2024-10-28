#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include "utils.hpp"
#include "class.hpp"

int main(int argc, char* argv[]) {
    if (argc > 1){
        NeuralNetwork nn;
        std::string command = argv[1];
        if (command == "create"){
            std::vector<int> input_shape = stois(argv[2]);
            std::vector<std::vector<int>> kernel_shape = stoiss(argv[3]);
            std::vector<int> npc = stois(argv[4]);
            nn.create(input_shape, kernel_shape, npc);
        }
        if (command == "train"){
            double learning_rate = std::stof(argv[2]);
            int n = std::stoi(argv[3]);
            int nb = std::stoi(argv[4]);
            int nbbt = std::stoi(argv[5]);
            nn.train(learning_rate, n, nb, nbbt);
        }
        else{
            throw std::invalid_argument("Invalid command");
            }
    }
    else{
        throw std::invalid_argument("No command");
    }
    return 0;
}