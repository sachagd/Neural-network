#ifndef CLASS_H
#define CLASS_H

#include <vector>

class C_layers{
    public :
        C_layers(std::pair<double*, std::vector<int>> kernels, std::pair<double*, std::vector<int>> biases);
        
        double * forward(const double * inputs, int n);

        double * backward(double * input_gradient, int n, double learning_rate);

        double * kernels;
        std::vector<int> kernels_shape;
        int kernels_size;
        double * biases;
        std::vector<int> biases_shape;
        int biases_size;

    private :
        int kernels_subsize;
        const double * inputs;
        std::vector<int> input_subshape;
        int input_subsize;
        int output_subsize;
        double * output_not_activated;  
};

class Fc_firstlayers{
    public :
        Fc_firstlayers(std::pair<double*, std::vector<int>> weights, std::pair<double*, std::vector<int>> biases);

        double * forward(double * inputs, int n);

        double * backward(double * input_gradient, int n, double learning_rate);

        double * weights;
        std::vector<int> weights_shape;
        int weights_size;
        double * biases;
        std::vector<int> biases_shape;
        int biases_size;

    private:
        double * inputs;
        double * output_not_activated;
};

class Fc_lastlayer{
    public :
        Fc_lastlayer();

        Fc_lastlayer(std::pair<double*, std::vector<int>> weights, std::pair<double*, std::vector<int>> biases);

        double * forward(double * inputs, int n);

        double * backward(double * y_pred, double * y_true, int n, double learning_rate);

        double * weights;
        std::vector<int> weights_shape;
        int weights_size;
        double * biases;
        std::vector<int> biases_shape;
        int biases_size;

    private:
        double * inputs;
};

class NeuralNetwork{
    public : 
        void create(std::vector<int> input_shape, std::vector<std::vector<int>> kernel_shape, std::vector<int> fcnpc);

        void train(double learning_rate, int n, int nb, int nbbt);

    private :
        void open();

        double * forpropagation(double * input, int n);

        void backpropagation(double * outputs, double * labels, int n, double learning_rate);

        double accuracy(double * acc_outputs, double * test_images, int * test_labels);

        std::vector<C_layers> cnn;
        std::vector<Fc_firstlayers> fcnnfl;
        Fc_lastlayer fcnnll;
        std::pair<std::vector<std::vector<int>>, int> info_1;
        std::pair<std::vector<int>, int> info_2;      
};

#endif