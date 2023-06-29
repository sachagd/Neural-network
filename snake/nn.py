from numpy import *
from tqdm import *
from random import randint
from matplotlib import *
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import os

class First_layers: # fct d'activation : Relu
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self,inputs):   
        self.inputs = inputs
        self.output_not_activated = np.dot(self.weights, inputs) + self.biases
        self.output = np.maximum(0, self.output_not_activated)

    def backward(self, input_gradient, learning_rate):
        input_gradient = input_gradient * deriv_relu(self.output_not_activated)
        self.output_gradient = np.dot(self.weights.T, input_gradient)
        self.weights -= np.dot(input_gradient, self.inputs.T) * learning_rate
        self.biases -= input_gradient * learning_rate 

class Last_layer: # fct d'activation : Softmax
    def __init__(self, weights, biases): 
        self.weights = weights
        self.biases = biases

    def forward(self,inputs): 
        self.inputs = inputs
        output_not_activated = np.dot(self.weights, inputs) + self.biases
        exp_values = np.exp(output_not_activated - np.max(output_not_activated, axis = 0, keepdims = True))
        self.output = exp_values / np.sum(exp_values, axis = 0, keepdims = True) 
    
    def backward(self, y_pred, y_true, learning_rate): # fct de perte : Categorical-crossentropy
        input_gradient = y_pred - y_true
        self.output_gradient = np.dot(self.weights.T, input_gradient)
        self.weights -= np.dot(input_gradient, self.inputs.T) * learning_rate
        self.biases -= input_gradient * learning_rate

def open():
    global nn
    weights = np.load("weights.npy", [], True)
    biases = np.load("biases.npy", [], True)
    n = len(weights)
    nn = [First_layers(weights[j], biases[j]) for j in range(n-1)] + [Last_layer(weights[n-1], biases[n-1])]

def create(npc): # npc : neurones par couche
    n = len(npc)
    info = np.array([npc, 0, [0]], dtype = object)
    biases = np.array([np.random.randn(npc[j+1],1) for j in range(n-1)], dtype = object)
    weights = np.array([0.1 * np.random.randn(npc[j+1],npc[j]) for j in range(n-1)], dtype = object)
    np.save("info.npy", info)
    np.save("weights.npy", weights)
    np.save("biases.npy", biases)

def get_info():
    info = np.load("info.npy", [], True)
    print(info[0])
    plt.plot(range(len(info[2])), info[2]) 
    plt.title("best score : "+str(info[1]))
    plt.show()

def save():
    try:
        info = np.load("info.npy", [], True) 
        folder = str(round(info[1],2))+" "+str(info[0])
        os.makedirs(folder)
        os.rename("biases.npy",folder+"/biases.npy")
        os.rename("weights.npy",folder+"/weights.npy")
        os.rename("info.npy",folder+"/info.npy")
    except OSError:
        raise FileExistsError("pas de fichiers à sauvergarder")

def delete_file():
    try:
        os.remove("biases.npy")
        os.remove("weights.npy")
        os.remove("info.npy")
    except OSError:
        raise FileExistsError("pas de fichiers à supprimer")
    
def delete_folder(n):
    folders = os.listdir()
    folders.sort()
    folder = folders[n]
    b = input("le nom du dossier est "+folder+" êtes-vous sûr de continuer ? (o/n)")
    if b == "o":
        os.remove(folder+"/biases.npy")
        os.remove(folder+"/weights.npy")
        os.remove(folder+"/info.npy")
        os.rmdir(folder)

def load(n):
    folders = os.listdir()
    folders.sort()
    folder = folders[n]
    os.rename(folder+"/biases.npy", "biases.npy")
    os.rename(folder+"/weights.npy", "weights.npy")
    os.rename(folder+"/info.npy", "info.npy")

def resave(n, rename_folder = False):
    info = np.load("info.npy", [], True)
    folders = os.listdir()
    folders.sort()
    folder = folders[n]
    if len(os.listdir(folder)) == 0:
        if rename_folder:
            new_folder = str(round(info[1]*100,2))+" "+str(info[0])
            os.rename(folder, new_folder)
            folder = new_folder
        os.rename("biases.npy",folder+"/biases.npy")
        os.rename("weights.npy",folder+"/weights.npy")
        os.rename("info.npy",folder+"/info.npy")
    else:
        raise FileExistsError("des fichiers se trouvent déjà dans ce dossier")

def ls():
    for file in os.listdir():
        if file[0] in "1234567890":
            print(file)

def backpropagation(y_pred, y_true, learning_rate): 
    nn[-1].backward(y_pred, y_true, learning_rate)
    for i in range(len(nn)-2,-1,-1):
        nn[i].backward(nn[i+1].output_gradient, learning_rate)
    
def forpropagation(input):
    nn[0].forward(input)
    for i in range(1,len(nn)):
        nn[i].forward(nn[i-1].output)
    return nn[-1].output

def deriv_relu(x):
    return x>0

def train():
    open()
    snake(1)
    
def test(te = 1):
    global t
    t = te
    if te == 1:
        open()
    snake(0)

def snake(mode):
    global d, lm, add, score, t
    root = Tk()
    canvas = Canvas(root, background="black", width=420, height=520)
    d = 0
    lm = 0
    add = 0
    score = 0
    for i in range(0,421,20):
        canvas.create_line((0, i), (420, i), fill='grey25', width=1)
        canvas.create_line((i, 0), (i, 420), fill='grey25', width=1)
    snake = [[4,10,canvas.create_rectangle((80,200),(100,220), fill = "blue", outline = "blue")],[5,10,canvas.create_rectangle((100,200),(120,220), fill = "blue", outline = "blue")]]
    apple = [15, 10, canvas.create_rectangle((300,200),(320,220), fill = "red", outline = "red")]
    root.focus_force()
    canvas.pack()
    
    def prob(l):
        for i in range(len(snake)):
            if l == snake[i][:2]:
                return True
        return l[0]>20 or l[1]>20 or l[0]<0 or l[1]<0

    def end():
        root.destroy()
        if mode:
            test(0)
        else:
            global t
            if t == 0:
                info = np.load("info.npy", [], True)
                info[2].append(score)
                if score > info[1]:
                    info[1] = score
                best_weights = np.array([(nn[j].weights).copy() for j in range(len(nn))], dtype = object)
                best_biases = np.array([(nn[j].biases).copy() for j in range(len(nn))], dtype = object)
                # folder = str(len(os.listdir()))
                # os.mkdir(folder)
                # os.rename("weights.npy",folder+"/weights.npy")
                # os.rename("biases.npy",folder+"/biases.npy")
                # os.rename("info.npy",folder+"/info.npy")
                np.save("weights.npy", best_weights)
                np.save("biases.npy", best_biases)
                np.save("info.npy", info)
                print("save")

    def move():
        global d, add, lm, score
        pos = snake[-1].copy()
        input = [[pos[0]<apple[0]],[pos[1]<apple[1]],[pos[0]>apple[0]],[pos[1]>apple[1]]]
        for i in range(-1,2):
            for j in range(-1,2):
                if i!=0 or j!=0:
                    input.append([prob([pos[0]+i,pos[1]+j])])
        input = np.array(input)
        output = forpropagation(input)
        nn_d = np.argmax(output)
        learning_rate = 0.0001
        print(np.argmax(output))
        if mode:
            if d == 0:
                pos[0] += 1
                lm = 0
            elif d == 1:
                pos[1] += 1
                lm = 1
            elif d == 2:
                pos[0] -= 1
                lm = 2
            elif d == 3:
                pos[1] -= 1
                lm = 3
        else:
            if nn_d == 0:
                if lm == 2:
                    end()
                    return
                else:
                    pos[0] += 1
                    lm = 0
            elif nn_d == 1:
                if lm == 3:
                    end()
                    return
                else:
                    pos[1] += 1
                    lm = 1
            elif nn_d == 2:
                if lm == 0:
                    end()
                    return
                else:
                    pos[0] -= 1
                    lm = 2
            elif nn_d == 3:
                if lm ==1:
                    end()
                    return
                else:
                    pos[1] -= 1
                    lm = 3
        if prob(pos[:2]):
            end()
            return
        pos[2] = canvas.create_rectangle((pos[0] * 20,pos[1] * 20),(pos[0] * 20 + 20, pos[1] * 20 + 20), fill = "blue", outline = "blue")
        snake.append(pos)
        if add == 0:
            canvas.delete(snake[0][2])
            del snake[0]
        else:
            add -= 1
        if pos[:2] == apple[:2]:
            learning_rate = 0.0005
            if len(snake) == 441:
                end()
                return
            add += 1
            score +=1
            new_co1 = [randint(0,20),randint(0,20)]
            while prob(new_co1):
                new_co1 = [randint(0,20),randint(0,20)]
            new_co2 = [new_co1[0] * 20, new_co1[1] * 20, new_co1[0] * 20 + 20, new_co1[1] * 20 + 20]
            canvas.coords(apple[2], new_co2)
            apple[:2] = new_co1
        if not mode:
            root.after(100,move)
        else:
            backpropagation(output,np.array([[d==i] for i in range(4)]),learning_rate)

    def right(event):
        global d, lm
        if lm != 2:
            d = 0
            if mode:
                move()

    def left(event):
        global d, lm
        if lm != 0:
            d = 2
            if mode:
                move()

    def up(event):
        global d, lm
        if lm != 1 :
            d = 3
            if mode:
                move()

    def down(event):
        global d, lm
        if lm != 3:
            d = 1
            if mode:
                move()

    if not mode:
        move()
    else:
        bg = LabelFrame(root, bg="white")
        bg.place(x=140, y=450)
        button = Button(bg, text = "quit", height=2, width=20, command=end)
        button.pack()
        root.bind("<Left>", left)
        root.bind("<Right>", right)
        root.bind("<Up>", up)
        root.bind("<Down>", down)
    root.mainloop()