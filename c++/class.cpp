#include </home/sacha/cnn c++/numcpp.cpp>
#include <iostream>
#include <filesystem>
#include <vector>

class C_layers{
    public :
        C_layers(std::pair<double*, std::vector<int>> kernels, std::pair<double*, std::vector<int>> biases){
            this->kernels = kernels.first;
            this->kernels_shape = kernels.second;
            this->kernels_size = multiplication(this->kernels_shape);
            this->kernels_subsize = this->kernels_shape[2] * this->kernels_shape[3];
            this->biases = biases.first;
            this->biases_shape = biases.second;
            this->biases_size = multiplication(this->biases_shape);
            this->input_subshape = {this->biases_shape[1] + this->kernels_shape[2] - 1, this->biases_shape[2] + this->kernels_shape[3] - 1};
            this->input_subsize = multiplication(input_subshape);
            this->output_subsize = this->biases_shape[1] * this->biases_shape[2];
        }
        
        double * forward(const double * inputs, int n){
            this->inputs = inputs;
            this->output_not_activated = tile(this->biases, this->biases_size, n);
            const double * ptrInputs = inputs;
            double * baseptrOutputs = this->output_not_activated;
            double * ptrOutputs;
            double * ptrKernels;
            for (int i = 0; i < n; i++){
                ptrKernels = this->kernels;
                for (int j = 0; j < this->kernels_shape[0]; j++){
                    ptrOutputs = baseptrOutputs;
                    for (int k = 0; k < this->kernels_shape[1]; k++){
                        add1(ptrOutputs, valid_corr(ptrInputs, this->input_subshape[0], this->input_subshape[1], ptrKernels , this->kernels_shape[2], this->kernels_shape[3]), this->biases_shape[1] * this->biases_shape[2]);
                        ptrKernels += this->kernels_subsize;
                        ptrOutputs += this->output_subsize;
                    }
                    ptrInputs += this->input_subsize;
                }
                baseptrOutputs += this->biases_size;
            }
            return this->output_not_activated;
            //add activation function
        }

        double * backward(double * input_gradient, int n, double learning_rate){
            double * kernels_gradient = new double[this->kernels_size];
            double * output_gradient = new double[n * this->kernels_shape[0] * this->input_subsize];
            double * ptrOutput_gradient = output_gradient;
            double * ptrKernels;
            double * ptrKernels_gradient;
            const double * ptrInputs = this->inputs;
            double * baseptrInput_gradient = input_gradient;
            double * ptrInput_gradient;
            for (int i = 0; i < n; i++){
                ptrKernels = this->kernels;
                ptrKernels_gradient = kernels_gradient;
                ptrInput_gradient = baseptrInput_gradient;
                for (int j = 0; j < this->kernels_shape[0]; j++){
                    for (int k = 0; k < this->kernels_shape[1]; k++){
                        add1(ptrKernels_gradient, valid_corr(ptrInputs, this->input_subshape[0], this->input_subshape[1], ptrInput_gradient, this->biases_shape[1], this->biases_shape[2]), this->kernels_subsize);
                        add1(ptrOutput_gradient, full_conv(ptrInput_gradient, this->biases_shape[1], this->biases_shape[2], ptrKernels, this->kernels_shape[2], this->kernels_shape[3]), this->input_subsize);
                        ptrKernels += this->kernels_subsize;
                        ptrKernels_gradient += this->kernels_subsize;
                        ptrInput_gradient += this->output_subsize;
                    } 
                    ptrOutput_gradient += this->input_subsize;
                    ptrInputs += this->input_subsize;
                }
                baseptrInput_gradient += this->biases_size;
            }
            sub1(this->kernels, mul2(kernels_gradient, learning_rate / n, this->kernels_size), this->kernels_size);
            sub1(this->biases, mul2(mean(input_gradient, n, this->biases_size), learning_rate, this->biases_size), this->biases_size);
            return output_gradient;
        }

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
        Fc_firstlayers(std::pair<double*, std::vector<int>> weights, std::pair<double*, std::vector<int>> biases){
            this->weights = weights.first;
            this->weights_shape = weights.second;
            this->weights_size = multiplication(this->weights_shape);
            this->biases =  biases.first;
            this->biases_shape = biases.second;
            this->biases_size = multiplication(this->biases_shape);
        }

        double * forward(double * inputs, int n){
            this->inputs = inputs;
            this->output_not_activated = add2(dot(inputs, this->weights, n, this->weights_shape[0], this->weights_shape[1]), this->biases, n, this->weights_shape[1]);
            return maximum(this->output_not_activated, 0, n * this->weights_shape[1]);
        }

        double * backward(double * input_gradient, int n, double learning_rate){
            mul_by_deriv_relu(input_gradient, this->output_not_activated, n * this->weights_shape[1]);
            double * output_gradient = dot(input_gradient, transpose(this->weights, this->weights_shape[0], this->weights_shape[1]), n, this->weights_shape[1], this->weights_shape[0]);
            sub1(this->weights, mul2(dot(transpose(this->inputs, n, this->weights_shape[0]), input_gradient, this->weights_shape[0], n, this->weights_shape[1]), learning_rate / n, this->weights_size), this->weights_size);
            sub1(this->biases, mul2(mean(input_gradient, n, this->weights_shape[1]), learning_rate, this->weights_shape[1]), this->weights_shape[1]);
            return output_gradient;
        }

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
        Fc_lastlayer() : weights(nullptr), biases(nullptr) {}

        Fc_lastlayer(std::pair<double*, std::vector<int>> weights, std::pair<double*, std::vector<int>> biases){
            this->weights = weights.first;
            this->weights_shape = weights.second;
            this->weights_size = multiplication(this->weights_shape);
            this->biases =  biases.first;
            this->biases_shape = biases.second;
            this->biases_size = multiplication(this->biases_shape);
        }

        double * forward(double * inputs, int n){
            this->inputs = inputs;
            double * output_not_activated = add2(dot(inputs, this->weights, n, this->weights_shape[0], this->weights_shape[1]), this->biases, n, this->weights_shape[1]);
            double * exp_values = exp(sub2(output_not_activated, max(output_not_activated, n, this->weights_shape[1]), n, this->weights_shape[1]), n * this->weights_shape[1]);
            return div(exp_values, sum(exp_values, n, this->weights_shape[1]), n, this->weights_shape[1]);
        }

        double * backward(double * y_pred, double * y_true, int n, double learning_rate){
            double * input_gradient = sub1(y_pred, y_true, n * 10);
            double * output_gradient = dot(input_gradient, transpose(this->weights, this->weights_shape[0], 10), n, 10, this->weights_shape[0]);
            sub1(this->weights, mul2(dot(transpose(this->inputs, n, this->weights_shape[0]), input_gradient, this->weights_shape[0], n, 10), learning_rate / n, this->weights_shape[0] * 10), this->weights_shape[0] * 10);
            sub1(this->biases, mul2(mean(input_gradient, n, 10), learning_rate, 10), 10);
            return output_gradient;
        }

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
        void create(std::vector<int> input_shape, std::vector<std::vector<int>> kernel_shape, std::vector<int> fcnpc){
            create_folders({"kernels", "cbiases", "weights", "fcbiases", "info"}, 5);
            int nfcnn = fcnpc.size();
            int ncnn = kernel_shape.size();
            for (int i = 0; i < ncnn; i++){
                input_shape = {input_shape[0] + 1 - kernel_shape[i][2], input_shape[1] + 1 - kernel_shape[i][3]};
                std::vector<int> cbiases_shape = {kernel_shape[i][1], input_shape[0], input_shape[1]};
                save_matrix(random_matrix(kernel_shape[i]), kernel_shape[i], 4, "kernels/kernels_" + std::to_string(i));
                save_matrix(random_matrix(cbiases_shape), cbiases_shape, 3, "cbiases/cbiases_" + std::to_string(i));
            }
            int reshape_shape = kernel_shape[ncnn - 1][1] * input_shape[0] * input_shape[1];
            save_matrix(random_matrix({reshape_shape, fcnpc[0]}), {reshape_shape, fcnpc[0]}, 2, "weights/weights_0");
            save_matrix(random_matrix({fcnpc[0]}), {fcnpc[0]}, 1, "fcbiases/fcbiases_0");
            for (int i = 0; i < nfcnn - 1; i++){
                save_matrix(random_matrix({fcnpc[i], fcnpc[i+1]}), {fcnpc[i], fcnpc[i+1]}, 2, "weights/weights_" + std::to_string(i + 1));
                save_matrix(random_matrix({fcnpc[i+1]}), {fcnpc[i+1]}, 1, "fcbiases/fcbiases_" + std::to_string(i + 1));
            }
            save_info_2d(kernel_shape, ncnn, "info/info_1");
            save_info_1d(fcnpc, nfcnn, "info/info_2");
            save_info_1d({}, 0, "info/info_3");
            save_info_1d({0}, 1, "info/info_4");
        }

        void train(double learning_rate, int n, int nb, int nbbt){

            open();

            std::random_device rd;
            std::mt19937 gen(rd());
            
            double * train_images = load_dataset_images("datasets/train_images", 47040000);
            int * train_labels = load_dataset_labels("datasets/train_labels", 60000);
            double * test_images = load_dataset_images("datasets/test_images", 7840000);
            int * test_labels = load_dataset_labels("datasets/test_labels", 10000);

            int ncnn = this->info_1.second;
            int nfcnn = this->info_2.second;
            int best_acc = load_info_1d("info/info_4").first[0];

            double * inputs = new double[n * 784];
            double * labels = new double[n * 10];
            double * ptrinputs;
            double * ptrlabels;
            double * outputs;
            double * acc_outputs;
            bool getting_better = false;

            std::vector<double *> best_kernels(ncnn);
            std::vector<double *> best_cbiases(ncnn);
            std::vector<double *> best_weights(nfcnn);
            std::vector<double *> best_fcbiases(nfcnn);

            for (int i = 0; i < nb; i++){
                for (int j = 0; j < nbbt; j++){
                    ptrinputs = inputs;
                    ptrlabels = labels;
                    for (int k = 0; k < n; k++){
                        int e = randint(gen);
                        memcpy(ptrinputs, train_images + 784 * e, 784 * sizeof(double));
                        int label = train_labels[e];
                        for (int l = 0; l < 10; l++){
                            if (l == label){
                                *(ptrlabels + l) = 1.;
                            }
                            else{
                                *(ptrlabels + l) = 0.;
                            }
                        }
                        ptrinputs += 784;
                        ptrlabels += 10;
                    }
                    outputs = forpropagation(inputs, n);
                    backpropagation(outputs, labels, n, learning_rate);
                }
                double acc = accuracy(acc_outputs, test_images, test_labels);
                if (acc > best_acc){
                    getting_better = true;
                    for (int j = 0; j < ncnn; j++){
                        memcpy(best_kernels[j], this->cnn[j].kernels, this->cnn[j].kernels_size * sizeof(double));
                        memcpy(best_cbiases[j], this->cnn[j].biases, this->cnn[j].biases_size * sizeof(double));
                    }
                    for (int j = 0; j < nfcnn - 1; j++){
                        memcpy(best_weights[j], this->fcnnfl[j].weights, this->fcnnfl[j].weights_size * sizeof(double));
                        memcpy(best_fcbiases[j], this->fcnnfl[j].biases, this->fcnnfl[j].biases_size * sizeof(double));
                    }
                    memcpy(best_weights[nfcnn - 1], this->fcnnll.weights, this->fcnnll.weights_size * sizeof(double));
                    memcpy(best_fcbiases[nfcnn - 1], this->fcnnll.biases, this->fcnnll.biases_size * sizeof(double));
                }
            }
            if (getting_better){
                for (int i = 0; i < ncnn; i++){
                    save_matrix(best_kernels[i], this->cnn[i].kernels_shape, 4, "kernels/kernels_" + std::to_string(i));
                    save_matrix(best_cbiases[i], this->cnn[i].biases_shape, 3, "cbiases/cbiases_" + std::to_string(i));
                }
                for (int i = 0; i < nfcnn - 1; i++){
                    save_matrix(best_weights[i], this->fcnnfl[i].weights_shape, 2, "kernels/kernels_" + std::to_string(i));
                    save_matrix(best_fcbiases[i], this->fcnnfl[i].biases_shape, 1, "fcbiases/fcbiases_" + std::to_string(i));
                }
                save_matrix(best_weights[nfcnn - 1], this->fcnnll.weights_shape, 2, "kernels/kernels_" + std::to_string(nfcnn - 1));
                save_matrix(best_fcbiases[nfcnn - 1], this->fcnnll.biases_shape, 1, "fcbiases/fcbiases_" + std::to_string(nfcnn - 1));
            }
            delete[] train_images;
            delete[] train_labels;
            delete[] test_images;
            delete[] test_labels;
            delete[] inputs;
            delete[] labels;
        }

    private :
        void open(){
            this->info_1 = load_info_2d("info/info_1");
            this->info_2 = load_info_1d("info/info_2");
            for (int i = 0; i < this->info_1.second; i++){
                this->cnn.push_back(C_layers(load_matrix("kernels/kernels_" + std::to_string(i)), load_matrix("cbiases/cbiases_" + std::to_string(i))));
            }
            for (int i = 0; i < this->info_2.second - 1; i++){
                this->fcnnfl.push_back(Fc_firstlayers(load_matrix("weights/weights_" + std::to_string(i)), load_matrix("fcbiases/fcbiases_" + std::to_string(i))));
            }
            this->fcnnll = Fc_lastlayer(load_matrix("weights/weights_" + std::to_string(info_2.second)), load_matrix("fcbiases/fcbiases_" + std::to_string(info_2.second)));
        }

        double * forpropagation(double * input, int n){
            double * outputs = this->cnn[0].forward(input, n);
            for (int i = 1; i < this->info_1.second; i++){
                outputs = this->cnn[i].forward(outputs, n);
            }
            for (int i = 0; i < this->info_2.second; i++){
                outputs = this->fcnnfl[i].forward(outputs, n);
            }
            outputs = this->fcnnll.forward(outputs, n);
            return outputs;
        }

        void backpropagation(double * outputs, double * labels, int n, double learning_rate){
            double * output_gradient = fcnnll.backward(outputs, labels, n, learning_rate);
            for (int i = this->info_2.second - 1; i > -1; i--){
                output_gradient = fcnnfl[i].backward(output_gradient, n, learning_rate);
            }
            for (int i = this->info_1.second - 1; i > -1; i--){
                output_gradient = cnn[i].backward(output_gradient, n, learning_rate);
            }
        }

        double accuracy(double * acc_outputs, double * test_images, int * test_labels){
            acc_outputs = forpropagation(test_images, 10000);
            int ind;
            double max;
            double acc;
            int baseacc_outputs = 0;
            for (int j = 0; j < 10000; j++){
                ind = 0;
                max = acc_outputs[baseacc_outputs];
                for (int k = 0; k < 10; k++){
                    if (max < acc_outputs[baseacc_outputs + k]){
                        max = acc_outputs[baseacc_outputs + k];
                        ind = k;
                    }
                }
                baseacc_outputs += 10;
                if (test_labels[j] == ind){
                    acc++;
                }
            }
            return acc / 10000;
        }

        std::vector<C_layers> cnn;
        std::vector<Fc_firstlayers> fcnnfl;
        Fc_lastlayer fcnnll;
        std::pair<std::vector<std::vector<int>>, int> info_1;
        std::pair<std::vector<int>, int> info_2;      
};