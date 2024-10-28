#include <random>
#include <filesystem>
#include <fstream>
#include "utils.hpp"

double random(std::mt19937& gen){
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    return dist(gen);
}

int randint(std::mt19937& gen){
    std::uniform_int_distribution<> dist(0, 59999);
    return dist(gen);
}

int multiplication(const std::vector<int> t){
    int r = 1;
    for (int i = 0; i < t.size(); i++){
        r *= t[i];
    }
    return r;
}

void create_folders(std::vector<std::string> folder_name, int n){
    for (int i = 0; i < n; i++){
        if (!std::filesystem::exists(folder_name[i])){
            std::filesystem::create_directory(folder_name[i]);
        }
    }
}

std::vector<int> stois(const std::string& arg) {
    std::string numbers = arg.substr(1, arg.size() - 2);
    std::stringstream ss(numbers);
    std::vector<int> intVector;
    std::string temp;
    while (std::getline(ss, temp, ',')) {
        intVector.push_back(std::stoi(temp));
    }
    return intVector;
}

std::vector<std::vector<int>> stoiss(const std::string& arg) {
    int size = 0;
    std::string trimmed = arg.substr(1, arg.size() - 2);
    std::vector<std::vector<int>> matrix;
    std::stringstream ss(trimmed);
    std::string segment;
    while (std::getline(ss, segment, ']')){
        if (segment[0] == ','){
            segment = segment.substr(1); 
        }
        matrix.push_back(stois(segment + ']'));
    }
    return matrix;
}

void save_info_1d(const std::vector<int> info, int n, const std::string& filename){
    std::ofstream outFile(filename, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(&n), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(info.data()), n * sizeof(int));
    outFile.close();
}

std::pair<std::vector<int>, int> load_info_1d(const std::string& filename){
    std::ifstream inFile(filename, std::ios::binary);
    int n;
    inFile.read(reinterpret_cast<char *>(&n), sizeof(int));
    std::vector<int> info(n);
    inFile.read(reinterpret_cast<char *>(info.data()), n * sizeof(int));
    inFile.close();
    return {info, n};
}

void save_info_2d(const std::vector<std::vector<int>> info, int n, const std::string& filename){
    std::ofstream outFile(filename, std::ios::binary);
    outFile.write(reinterpret_cast<const char *>(&n), sizeof(int));
    for(int i = 0; i < n; i++){
        outFile.write(reinterpret_cast<const char*>(info[i].data()), 4 * sizeof(int));
    }
    outFile.close();
}

std::pair<std::vector<std::vector<int>>, int> load_info_2d(const std::string& filename){
    std::ifstream inFile(filename, std::ios::binary);
    int n;
    inFile.read(reinterpret_cast<char *>(&n), sizeof(int));
    std::vector<std::vector<int>> info(n);
    for (int i = 0; i < n; i++){
        inFile.read(reinterpret_cast<char *>(info[i].data()), 4 * sizeof(int));
    }
    inFile.close();
    return {info, n};
}

double * load_dataset_images(const std::string& filename, int n){
    std::ifstream inFile(filename, std::ios::binary);
    double * dataset = new double[n];
    inFile.read(reinterpret_cast<char*>(dataset), n * sizeof(double));
    inFile.close();
    return dataset;
}

int * load_dataset_labels(const std::string& filename, int n){
    std::ifstream inFile(filename, std::ios::binary);
    int * dataset = new int[n];
    inFile.read(reinterpret_cast<char*>(dataset), n * sizeof(int));
    inFile.close();
    return dataset;
}