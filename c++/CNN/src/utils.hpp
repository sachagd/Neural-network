#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <filesystem>

double random(std::mt19937& gen);

int randint(std::mt19937& gen);

int multiplication(const std::vector<int> t);

void create_folders(std::vector<std::string> folder_name, int n);

std::vector<int> stois(const std::string& arg);

std::vector<std::vector<int>> stoiss(const std::string& arg);

void save_info_1d(const std::vector<int> info, int n, const std::string& filename);

std::pair<std::vector<int>, int> load_info_1d(const std::string& filename);

void save_info_2d(const std::vector<std::vector<int>> info, int n, const std::string& filename);

std::pair<std::vector<std::vector<int>>, int> load_info_2d(const std::string& filename);

double * load_dataset_images(const std::string& filename, int n);

int * load_dataset_labels(const std::string& filename, int n);

#endif