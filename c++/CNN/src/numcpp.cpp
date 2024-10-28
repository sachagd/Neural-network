#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "numcpp.hpp"
#include "utils.hpp"

void add_full_conv(double * matrix1, const double * matrix2, int mn, int mp , const double * kernel, int kn, int kp){
    int rn = mn + kn - 1;
    int rp = mp + kp - 1;
    int resultIndex = 0;
    for (int i = 0; i < rn; i++){
        for (int j = 0; j < rp; j++){
            double sum = 0;
            int kernelIndex = 0;
            int matrixIndex_x = i;
            int matrixIndex = i * mp + j; //optimiser cela ?
            for (int k = 0; k < kn; k++){
                int matrixIndex_y = j;
                for (int l = 0; l < kp; l++){
                    if (matrixIndex_x >= 0 && matrixIndex_x < mn && matrixIndex_y >= 0 && matrixIndex_y < mp){
                        sum += kernel[kernelIndex] * matrix2[matrixIndex];
                    }
                    kernelIndex ++;
                    matrixIndex_y --;
                    matrixIndex --;
                }
                matrixIndex_x --;
                matrixIndex += kp - mp;
            }
            matrix1[resultIndex++] = sum;
        }
    }
}

void add_valid_corr(double * matrix1, const double * matrix2, int mn, int mp, const double * kernel, int kn, int kp){
    int rn = mn - kn + 1;
    int rp = mp - kp + 1;
    int resultIndex = 0;
    int matrixBaseIndex = 0;
    for (int i = 0; i < rn; i++){
        for (int j = 0; j < rp; j++){
            double sum = 0;
            int kernelIndex = 0;
            int matrixIndex = matrixBaseIndex;
            for (int k = 0; k < kn; k++){
                for (int l = 0; l < kp; l++){
                    sum += kernel[kernelIndex++] * matrix2[matrixIndex++];
                }
                matrixIndex += rp - 1;
            }
            matrix1[resultIndex++] = sum;
            matrixBaseIndex ++;
        }
        matrixBaseIndex += kp - 1;
    }
}

double * transpose(double * matrix, int n, int p){
    int index;
    int indexr = 0;
    double * r = new double[n * p];
    for (int i = 0; i < p; i++) {
        index = i;
        for (int j = 0; j < n; j++) {
            r[indexr++] = matrix[index];
            index += p;
        }
    }
    return r;
}

double * dot1(const double * matrix1, double * matrix2, int n, int p, int q){
    matrix2 = transpose(matrix2, p, q);
    int baseindex1 = 0;
    int index1;
    int index2;
    int indexr = 0;
    double * r = new double[n * q];
    for (int i = 0; i < n; i++){
        index2 = 0;
        for (int j = 0; j < q; j++){
            index1 = baseindex1;
            double c = 0;
            for (int k = 0; k < p; k++){
                c += matrix1[index1++] * matrix2[index2++];
            }
            r[indexr++] = c;
        }
        baseindex1 = indexr;
    }
    return r;
}

double * dot2(const double * matrix1, double * matrix2, int n, int p, int q){
    int baseindex1 = 0;
    int index1;
    int index2;
    int indexr = 0;
    double * r = new double[n * q];
    for (int i = 0; i < n; i++){
        index2 = 0;
        for (int j = 0; j < q; j++){
            index1 = baseindex1;
            double c = 0;
            for (int k = 0; k < p; k++){
                c += matrix1[index1++] * matrix2[index2++];
            }
            r[indexr++] = c;
        }
        baseindex1 = indexr;
    }
    return r;
}

void add1(double * matrix1, double * matrix2, int size){
    for (int i = 0; i < size; i++){
        matrix1[i] = matrix1[i] + matrix2[i];
    }
}

double * add2(double * matrix, double * tab, int n, int p){
    int indexm = 0;
    int indext;
    for (int i = 0; i < n; i++){
        indext = 0;
        for (int j = 0; j < p; j++){
            matrix[indexm++] += tab[indext++];
        }
    }
    return matrix;
}

double * sub(double * matrix1, double * matrix2, int size){
    for (int i = 0; i < size; i++){
        matrix1[i] = matrix1[i] - matrix2[i];
    }
    return matrix1;
}

void mul_deriv_relu(double * matrix1, double * matrix2, int size){
    for (int i = 0; i < size; i++){
        matrix1[i] = matrix1[i] * (matrix2[i] > 0);
    }
}

double * sub_mul(double * matrix1, double * matrix2, double e, int size){
    for (int i = 0; i < size; i++){
        matrix1[i] = matrix1[i] - (matrix2[i] * e);
    }
    return matrix1;
}

double * sub_mul_mean(double * matrix1, double * matrix2, double e, int n, int p){
    int index;
    double s;
    for (int i = 0; i < p; i++){
        index = i;
        s = 0;
        for (int j = 0; j < n; j++){
            s += matrix2[index];
            index += p;
        }
        matrix1[i] -= s * e / n;
    }
    return matrix1;
}

double * softmax(double * matrix, int n, int p){
    int index1 = 0;
    int index2 = 0;
    int index3 = 0;
    double s;
    double m;
    for (int i = 0; i < n; i++){
        m = matrix[index1];
        for (int j = 0; j < p; j++){
            if (matrix[index1] > m){
                m = matrix[index1];
            }
            index1++;
        }
        s = 0;
        for (int j = 0; j < p; j++){
            matrix[index2] = exp(matrix[index2] - m);
            s += matrix[index2];
            index2++;
        }
        for (int j = 0; j < p; j++){
            matrix[index3++] /= s;
        }
    }
    return matrix;
}

double * mean(double * matrix, int n, int p){
    double * r = new double[p];
    int index;
    double s;
    for (int i = 0; i < p; i++){
        index = i;
        s = 0;
        for (int j = 0; j < n; j++){
            s += matrix[index];
            index += p;
        }
        r[i] = s / n;
    }
    return r;
}

double * maximum(double * matrix, double e, int size){
    for (int i = 0; i < size; i++){
        matrix[i] = std::max(matrix[i], e);
    }
    return matrix;
}

double * tile(double * input, int size, int n) {
    double * r = new double[size * n];
    int total_size = size * sizeof(double);
    double * ptr = r;
    for (int i = 0; i < n; i++) {
        memcpy(ptr, input, total_size);
        ptr += size;
    }
    return r;
}

double * random_matrix(const std::vector<int> shape){
    std::random_device rd;
    std::mt19937 gen(rd());
    int n = multiplication(shape);
    double * r = new double[n];
    for (int i = 0; i < n; i++){
        r[i] = random(gen);
    }
    return r;
}

void save_matrix(const double* matrix, const std::vector<int> shape, int numDimensions, const std::string& filename){
    std::ofstream outFile(filename, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(&numDimensions), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(shape.data()), numDimensions * sizeof(int));
    outFile.write(reinterpret_cast<const char*>(matrix), multiplication(shape) * sizeof(double));
    outFile.close();
}

std::pair<double*, std::vector<int>> load_matrix(const std::string& filename){
    std::ifstream inFile(filename, std::ios::binary);
    int numDimensions;
    inFile.read(reinterpret_cast<char*>(&numDimensions), sizeof(int));
    std::vector<int> shape(numDimensions);
    inFile.read(reinterpret_cast<char*>(shape.data()), numDimensions * sizeof(int));
    int numElements = multiplication(shape);
    double * matrix = new double[numElements];
    inFile.read(reinterpret_cast<char*>(matrix), numElements * sizeof(double));
    inFile.close();
    return {matrix, shape};
}