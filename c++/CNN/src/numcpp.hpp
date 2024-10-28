#ifndef NUMCPP_H
#define NUMCPP_H

#include <vector>
#include <string>

void add_full_conv(double * matrix1, const double * matrix2, int mn, int mp , const double * kernel, int kn, int kp);

void add_valid_corr(double * matrix1, const double * matrix2, int mn, int mp, const double * kernel, int kn, int kp);

double * transpose(double * matrix, int n, int p);

double * dot1(const double * matrix1, double * matrix2, int n, int p, int q);

double * dot2(const double * matrix1, double * matrix2, int n, int p, int q);
 
void add1(double * matrix1, double * matrix2, int size);

double * add2(double * matrix, double * tab, int n, int p);

double * sub(double * matrix1, double * matrix2, int size);

void mul_deriv_relu(double * matrix1, double * matrix2, int size);

double * sub_mul(double * matrix1, double * matrix2, double e, int size);

double * sub_mul_mean(double * matrix1, double * matrix2, double e, int n, int p);

double * softmax(double * matrix, int n, int p);

double * mean(double * matrix, int n, int p);

double * maximum(double * matrix, double e, int size);

double * tile(double * input, int size, int n);

double * random_matrix(const std::vector<int> shape);

void save_matrix(const double* matrix, const std::vector<int> shape, int numDimensions, const std::string& filename);

std::pair<double*, std::vector<int>> load_matrix(const std::string& filename);

#endif