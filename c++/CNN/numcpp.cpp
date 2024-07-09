#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include </home/sacha/cnn c++/utils.cpp>
#include <iostream>
#include <algorithm>
#include <cmath>

double * full_conv(const double * matrix, int mn, int mp, const double * kernel, int kn, int kp){
    int rn = mn + kn - 1;
    int rp = mp + kp - 1;
    double * r = new double[rn * rp];
    int resultIndex = 0;
    for (int i = 0; i < rn; i++){
        for (int j = 0; j < rp; j++){
            double sum = 0;
            int kernelIndex = 0;
            int matrixIndex_x = i;
            int matrixIndex = i * mp + j;
            for (int k = 0; k < kn; k++){
                int matrixIndex_y = j;
                for (int l = 0; l < kp; l++){
                    if (matrixIndex_x >= 0 && matrixIndex_x < mn && matrixIndex_y >= 0 && matrixIndex_y < mp){
                        sum += kernel[kernelIndex] * matrix[matrixIndex];
                    }
                    kernelIndex ++;
                    matrixIndex_y --;
                    matrixIndex --;
                }
                matrixIndex_x --;
                matrixIndex += kp - mp;
            }
            r[resultIndex++] = sum;
        }
    }
    return r;
}

double * valid_corr(const double * matrix, int mn, int mp, const double * kernel, int kn, int kp){
    int rn = mn - kn + 1;
    int rp = mp - kp + 1;
    double * r = new double[rn * rp];
    int resultIndex = 0;
    int matrixBaseIndex = 0;
    for (int i = 0; i < rn; i++){
        for (int j = 0; j < rp; j++){
            double sum = 0;
            int kernelIndex = 0;
            int matrixIndex = matrixBaseIndex;
            for (int k = 0; k < kn; k++){
                for (int l = 0; l < kp; l++){
                    sum += kernel[kernelIndex++] * matrix[matrixIndex++];
                }
                matrixIndex += rp - 1;
            }
            r[resultIndex++] = sum;
            matrixBaseIndex ++;
        }
        matrixBaseIndex += kp - 1;
    }
    return r;
}

double * transpose(double * matrix, int n, int p){
    int index = 0;
    int indexr;
    double * r = new double[n * p];
    for (int i = 0; i < n; i++){
        indexr = i;
        for (int j = 0; j < p; j++){
            r[indexr] = matrix[index++];
            indexr += n;
        }
    } 
    return r;
} // modif pour la rendre en place

double * dot(const double * matrix1, double * matrix2, int n, int p, int q){
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

double * sub1(double * matrix1, double * matrix2, int size){
    for (int i = 0; i < size; i++){
        matrix1[i] = matrix1[i] - matrix2[i];
    }
    return matrix1;
}

double * sub2(double * matrix, double * tab, int n, int p){
    int index = 0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < p; j++){
            matrix[index++] -= tab[i];
            index++;
        }
    }
    return matrix;
}

void mul_by_deriv_relu(double * matrix1, double * matrix2, int size){
    for (int i = 0; i < size; i++){
        matrix1[i] = matrix1[i] * (matrix2 > 0);
    }
}

double * div(double * matrix, double * tab, int n, int p){
    int index = 0;
    for (int i = 0; i < n; i++){
        for (int j = 0; j < p; j++){
            matrix[index] = matrix[index] / tab[i];
            index++;
        }
    }
    return matrix;
}

double * mul2(double * matrix, double e, int size){
    for (int i = 0; i < size; i++){
        matrix[i] = matrix[i] * e;
    }
    return matrix;
}

double * exp(double * matrix, int size){
    for (int i = 0; i < size; i++){
        matrix[i] = exp(matrix[i]);
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

double * sum(double * matrix, int n, int p){
    double * r = new double[n];
    int index = 0;
    for (int i = 0; i < n; i++){
        r[i] = 0;
        for (int j = 0; j < p; j++){
           r[i] += matrix[index++];
        }
    }
    return r;
}

double * maximum(double * matrix, double e, int size){
    for (int i = 0; i < size; i++){
        matrix[i] = std::max(matrix[i], e);
    }
    return matrix;
}

double * max(double * matrix, int n, int p){
    double * r = new double[n];
    int index = 0;
    for (int i = 0; i < n; i++){
        double m = matrix[index];
        for (int j = 1; j < p; j++){
            if (matrix[index] > m){
                m = matrix[index];
            }
            index++;
        }
        r[i] = m;
    }
    return r;
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
