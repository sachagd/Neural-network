from numpy import *
from scipy.signal import correlate2d, convolve2d
from random import randint
from tqdm import *
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pyautogui

class Layers_c: #fct d'activation (not really) : max_pooling (not coded)
    def __init__(self, kernels, biases, kernels_shape):
        self.kernels = kernels 
        self.biases = biases
        self.kernels_shape = kernels_shape

    def forward(self, inputs):
        self.inputs = inputs
        self.inputs_shape = np.shape(inputs)
        self.output_not_activated = np.tile(self.biases, (np.shape(inputs)[0], 1, 1, 1))
        for i in range(self.inputs_shape[0]):
            for j in range(self.kernels_shape[0]):
                for k in range(self.kernels_shape[1]):
                    self.output_not_activated[i, j] += correlate2d(inputs[i, k], self.kernels[j, k], "valid")
        self.output = self.output_not_activated 

    def backward(self, input_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        self.output_gradient = np.zeros(self.inputs_shape)
        for i in range(self.inputs_shape[0]):
            for j in range(self.kernels_shape[0]):
                for k in range(self.kernels_shape[1]):
                    kernels_gradient[j, k] += correlate2d(self.inputs[i, k], input_gradient[i, j], "valid")
                    self.output_gradient[i, k] += convolve2d(input_gradient[i, j], self.kernels[j, k], "full")
        self.kernels -= kernels_gradient * learning_rate * 1/self.inputs_shape[0]
        self.biases -= np.mean(input_gradient, axis = 0) * learning_rate

class Reshape:
    def __init__(self, shape):
        self.input_shape = shape
        self.output_shape = np.prod(shape)

    def forward(self, input):
        return np.reshape(input, (np.shape(input)[0], self.output_shape))

    def backward(self, output_gradient):
        return np.reshape(output_gradient, (np.shape(output_gradient)[0], *self.input_shape))

class First_layers_fc: # fct d'activation : Relu
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs):   
        self.inputs = inputs
        self.output_not_activated = np.dot(inputs, self.weights) + self.biases
        self.output = np.maximum(0, self.output_not_activated)

    def backward(self, input_gradient, learning_rate):
        input_gradient = input_gradient * deriv_relu(self.output_not_activated)
        self.output_gradient = np.dot(input_gradient, self.weights.T)
        self.weights -= np.dot(self.inputs.T, input_gradient) * learning_rate * 1/np.shape(input_gradient)[0]
        self.biases -= np.mean(input_gradient, axis = 0) * learning_rate 

class Last_layer_fc: # fct d'activation : Softmax
    def __init__(self, weights, biases): 
        self.weights = np.array(weights, dtype = np.float64)
        self.biases = np.array(biases, dtype = np.float64)

    def forward(self, inputs): 
        self.inputs = inputs
        output_not_activated = np.dot(inputs, self.weights) + self.biases
        exp_values = np.exp(output_not_activated - np.max(output_not_activated, axis = 1, keepdims = True))
        self.output = exp_values / np.sum(exp_values, axis = 1, keepdims = True) 

    def backward(self, y_pred, y_true, learning_rate): # fct de perte : Categorical-crossentropy
        input_gradient = y_pred - y_true 
        self.output_gradient = np.dot(input_gradient, self.weights.T)
        self.weights -= np.dot(self.inputs.T, input_gradient) * learning_rate * 1/np.shape(input_gradient)[0]
        self.biases -= np.mean(input_gradient, axis = 0) * learning_rate

def open():
    global cnn, reshape, fcnn, acc_input, acc_label, dstest
    kernels = np.load("kernels.npy", [], True)
    cbiases = np.load("cbiases.npy", [], True)
    weights = np.load("weights.npy", [], True)
    fcbiases = np.load("fcbiases.npy", [], True)
    dstest = np.load("test_cnn.npy", [], True)
    n1 = np.shape(weights)[0]
    n2 = np.shape(kernels)[0]
    acc_input = np.array([dstest[i][1] for i in range(1000)])
    acc_label = np.array([dstest[i][0] for i in range(1000)])
    cnn = [Layers_c(kernels[i], cbiases[i], np.shape(kernels[i])) for i in range(n2)]
    reshape = Reshape(np.shape(cbiases[-1]))
    fcnn = [First_layers_fc(weights[j], fcbiases[j]) for j in range(n1-1)] + [Last_layer_fc(weights[n1-1], fcbiases[n1-1])]

def create(input_shape, kernel_shape, fcnpc):
    n1 = len(kernel_shape)
    n2 = len(fcnpc)
    info = np.array([kernel_shape+fcnpc, 0, [0]], dtype = object)
    cbiases = []
    kernels = []
    for i in range(n1):
        input_shape = [input_shape[0]-kernel_shape[i][2]+1, input_shape[1]-kernel_shape[i][3]+1]
        cbiases.append(np.random.randn(kernel_shape[i][0], *input_shape))
        kernels.append(np.random.randn(*kernel_shape[i]))
    fcnpc.insert(0, kernel_shape[n1-1][0] * input_shape[0] * input_shape[1]) 
    fcbiases = np.array([np.random.randn(fcnpc[j+1]) for j in range(n2)], dtype = object)
    weights = np.array([0.1 * np.random.randn(fcnpc[j],fcnpc[j+1]) for j in range(n2)], dtype = object)
    np.save("info.npy", info)
    np.save("kernels.npy", np.array(kernels, dtype = object))
    np.save("cbiases.npy", np.array(cbiases, dtype = object))
    np.save("weights.npy", weights)
    np.save("fcbiases.npy", fcbiases)

def get_info():
    info = np.load("info.npy", [], True)
    print(info[0])
    plt.plot(range(len(info[2])), info[2]) 
    plt.title("best accuracy : "+str(info[1]))
    plt.show()

def save():
    try:
        info = np.load("info.npy", [], True) 
        folder = f"{len(os.listdir())-8} {round(info[1]*100,2)} {info[0]}"
        os.makedirs(folder)
        os.rename("fcbiases.npy",folder+"/fcbiases.npy")
        os.rename("weights.npy",folder+"/weights.npy")
        os.rename("cbiases.npy",folder+"/cbiases.npy")
        os.rename("kernels.npy",folder+"/kernels.npy")
        os.rename("info.npy",folder+"/info.npy")
    except OSError:
        raise FileExistsError("pas de fichiers à sauvergarder")

def delete_file():
    try:
        os.remove("fcbiases.npy")
        os.remove("weights.npy")
        os.remove("cbiases.npy")
        os.remove("kernels.npy")
        os.remove("info.npy")
    except OSError:
        raise FileExistsError("pas de fichiers à supprimer")

def delete_folder(n):
    folder = os.listdir()[n]
    b = input(f"le nom du dossier est {folder} êtes-vous sûr de continuer ? (o/n)")
    if b == "o":
        os.remove(folder+"/fcbiases.npy")
        os.remove(folder+"/weights.npy")
        os.remove(folder+"/cbiases.npy")
        os.remove(folder+"/kernels.npy")
        os.remove(folder+"/info.npy")
        os.rmdir(folder)

def load(n):
    for folder in os.listdir():
        if os.path.isdir(folder) and folder.startswith(str(n)):
            os.rename(folder+"/fcbiases.npy", "fcbiases.npy")
            os.rename(folder+"/weights.npy", "weights.npy")
            os.rename(folder+"/cbiases.npy", "cbiases.npy")
            os.rename(folder+"/kernels.npy", "kernels.npy")
            os.rename(folder+"/info.npy", "info.npy")
            return
    raise FileExistsError(f"aucun dossier ne commence par {n}")

    
def resave(rename_folder = False):
    info = np.load("info.npy", [], True)
    for folder in os.listdir():
        if os.path.isdir(folder) and len(os.listdir(folder)) == 0:
            if rename_folder:
                id = re.match(r'^\d+\s', folder).group()
                new_folder = f"{id} {round(info[1] * 100, 2)} {info[0]}"
                os.rename(folder, new_folder)
            else:
                new_folder = folder
            os.rename("fcbiases.npy",new_folder+"/fcbiases.npy")
            os.rename("weights.npy",new_folder+"/weights.npy")
            os.rename("cbiases.npy",new_folder+"/cbiases.npy")
            os.rename("kernels.npy",new_folder+"/kernels.npy")
            os.rename("info.npy",new_folder+"/info.npy")
            return
    raise FileExistsError("pas de fichiers à sauvergarder")

def ls():
    for file in os.listdir():
        if file[0] in "1234567890":
            print(file)

def test(nb, only_false = False): 
    open()
    for i in range(nb):
        choix = randint(0,9999)
        input = np.array([dstest[choix][1]])
        label = dstest[choix][0]
        output = forpropagation(input)
        if not only_false or np.argmax(output)!=label:
            plt.gray()
            plt.imshow(input[0][0])
            plt.title(f"prediction :  {np.argmax(output)} label : {label}")
            plt.show(block=False)
            plt.pause(3)
    plt.close()

def train(lr_c, lr_fc , t_samples, nb, nbbt = 1):
    dstrain = np.load("train_cnn.npy", [], True)
    info = np.load("info.npy", [], True)
    best_acc = info[1]
    open()
    for _ in tqdm(range(nb)):
        for _ in tqdm(range(nbbt), leave=False):
            choix = [randint(0,59999) for _ in range(t_samples)] 
            inputs = np.array([dstrain[choix[i]][1] for i in range(t_samples)])
            labels = np.eye(10)[[dstrain[choix[i]][0] for i in range(t_samples)]]
            output = forpropagation(inputs)
            backpropagation(output, labels, lr_c, lr_fc)
        acc = accuracy()
        if acc == 0 or best_acc - acc > 0.3:
            raise ValueError("l'accuracy a chuté, diminuer le learning rate pourrait régler le problème")
        info[2].append(acc)
        if acc>best_acc:
            tqdm.write(f"l'accuracy a augmenté, la voici : {acc}")
            best_acc = acc
            best_weights = np.array([(fcnn[j].weights).copy() for j in range(len(fcnn))], dtype = object)
            best_fcbiases = np.array([(fcnn[j].biases).copy() for j in range(len(fcnn))], dtype = object)
            best_kernels = np.array([(cnn[j].kernels).copy() for j in range(len(cnn))], dtype = object)
            best_cbiases = np.array([(cnn[j].biases).copy() for j in range(len(cnn))], dtype = object)
        else:
            tqdm.write(f"l'accuracy n'a pas augmenté, la voici : {acc}")
    info[1] = best_acc
    np.save("info.npy", info)
    try :
        np.save("weights.npy", best_weights)
        np.save("fcbiases.npy", best_fcbiases)
        np.save("kernels.npy", best_kernels)
        np.save("cbiases.npy", best_cbiases)
    except UnboundLocalError:
        raise ValueError("pas de meilleure accuracy trouvé")

def accuracy():
    output = forpropagation(acc_input)
    return np.mean(np.argmax(output, axis = 1) == acc_label)

def total_accuracy():
    input = np.array([dstest[i][1] for i in range(10000)])
    label = np.array([dstest[i][0] for i in range(10000)])
    output = forpropagation(input)
    return np.mean(np.argmax(output, axis = 1) == label)

def forpropagation(input):
    cnn[0].forward(input)
    for i in range(1,len(cnn)):
        cnn[i].forward(cnn[i-1].output)
    fcnn[0].forward(reshape.forward(cnn[-1].output))
    for i in range(1,len(fcnn)):
        fcnn[i].forward(fcnn[i-1].output)
    return fcnn[-1].output

def backpropagation(y_pred, y_true, lr_c, lr_fc):
    fcnn[-1].backward(y_pred, y_true, lr_fc)
    for i in range(len(fcnn)-2,-1,-1):
        fcnn[i].backward(fcnn[i+1].output_gradient, lr_fc)
    cnn[-1].backward(reshape.backward(fcnn[0].output_gradient), lr_c)
    for i in range(len(cnn)-2,-1,-1):
        cnn[i].backward(cnn[i+1].output_gradient, lr_c)

def deriv_relu(x):
    return x>0

# def dessin():
#     global mode, condition
#     open()
#     root = Tk()
#     draw = [[0 for i in range(28)] for i in range(28)]
#     input = [[0 for i in range(28)] for i in range(28)]
#     canvas = Canvas(root, background="black", width=700, height=800)
#     text = canvas.create_text((200,745), text = str([0]*10) + " none", fill = "white")
#     for i in range(0,701,25):
#         canvas.create_line((0, i), (700, i), fill='grey25', width=1)
#         canvas.create_line((i, 0), (i, 700), fill='grey25', width=1)
#     root.focus_force()
#     canvas.pack()
#     condition = False
#     mode = "pencil"

#     def pointeur():
#         if condition:
#             xy = [xycan[0] + (pyautogui.position()[0] - xyscr[0]), xycan[1] + (pyautogui.position()[1]-xyscr[1])]
#             if (0 < xy[0] - xy[0]%25 + 25 < 701 and 0 < xy[1] - xy[1]%25 + 25 < 701) and (mode == "eraser" or (mode == "pencil" and input[xy[1]//25][xy[0]//25] != 255)):
#                 n = 2
#                 for i in range(-n,n+1):
#                     for j in range(-n,n+1):
#                         if sqrt(i**2 + j**2)<=n:
#                             if 0 < xy[0] - xy[0]%25 + 25 + i*25 < 701 and 0 < xy[1] - xy[1]%25 + 25 + j*25 < 701:
#                                 if mode == "pencil":
#                                     c = min(int((1 - (sqrt(i**2 + j**2)/n)) * 255) + input[xy[1]//25 + j][xy[0]//25 + i], 255)
#                                     rgb = "#%02x%02x%02x" % (c,c,c)
#                                     if c != 0:
#                                         input[xy[1]//25 + j][xy[0]//25 + i] = c
#                                         if draw[xy[1]//25 + j][xy[0]//25 + i] == 0:
#                                             draw[xy[1]//25 + j][xy[0]//25 + i] = canvas.create_rectangle((xy[0] - xy[0]%25 + i*25, xy[1] - xy[1]%25 + j*25), (xy[0] - xy[0]%25+25 + i*25, xy[1] - xy[1]%25 + 25 + j*25), fill = rgb, outline = rgb)
#                                         else:
#                                             canvas.itemconfig(draw[xy[1]//25 + j][xy[0]//25 + i], fill = rgb)
#                                             canvas.itemconfig(draw[xy[1]//25 + j][xy[0]//25 + i], outline = rgb)
#                                 if mode == "eraser" and draw[xy[1]//25 + j][xy[0]//25 + i] != 0:
#                                     canvas.delete(draw[xy[1]//25 + j][xy[0]//25 + i])
#                                     draw[xy[1]//25 + j][xy[0]//25 + i]=0
#                                     input[xy[1]//25 + j][xy[0]//25 + i]=0          
#                 input1 = np.reshape(np.array(input), (784,1))
#                 pred = forpropagation(input1)
#                 canvas.itemconfig(text, text = str(np.ndarray.tolist(np.reshape(np.round(pred, 1),(10,))))+ " " + str(np.argmax(pred)))
#         root.after(1, pointeur)

#     def beg(event):
#         global xycan,condition,xyscr
#         xycan,condition,xyscr = [event.x,event.y],True,pyautogui.position()

#     def end(event):
#         global condition
#         condition = False

#     def pencil():
#         global mode
#         bg2.configure(bg = "white")
#         bg1.configure(bg = "green")
#         mode = "pencil"

#     def eraser():
#         global mode
#         bg2.configure(bg = "green")
#         bg1.configure(bg = "white")
#         mode = "eraser"

#     def reset():
#         for i in range(28):
#             for j in range(28):
#                 if draw[i][j] != 0:
#                     canvas.delete(draw[i][j])
#                     draw[i][j] = 0
#                     input[i][j] = 0
#         canvas.itemconfig(text, text = str([0]*10) + " none")

#     bg1 = LabelFrame(root, bd=6, bg="green")
#     bg1.place(x=400,y=28*25+15)
#     icon1 = PhotoImage(file='pencil.png')
#     button1 = Button(bg1, height=50, width=50, image=icon1, command=pencil)
#     button1.pack()

#     bg2 = LabelFrame(root, bd=6, bg="white")
#     bg2.place(x=500, y=28*25+15)
#     icon2 = PhotoImage(file='eraser.png')
#     button2 = Button(bg2, height=50, width=50, image=icon2, command=eraser)
#     button2.pack()

#     bg3 = LabelFrame(root, bd=6, bg="white")
#     bg3.place(x=600, y=28*25+22)
#     button3 = Button(bg3, height=1, width=5, text="delete", command=reset)
#     button3.pack()

#     pointeur()
#     canvas.bind("<Button-1>", beg)
#     canvas.bind("<ButtonRelease-1>", end)
#     root.mainloop()
