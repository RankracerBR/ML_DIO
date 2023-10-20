from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, optim
from time import time
import numpy as np
import torchvision
import torch

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28,128) # Camada de entrada, 784 neurônios que se ligam a 128
        self.linear2 = nn.Linear(128,64) # Camada Interna 1, 128 neurônio que se ligam a 64
        self.linear3 = nn.Linear(64,10) # Camada Interna 2, 64 neurônios que se ligam a 10
        
    def forward(self, X):
        X = F.relu(self.linear1(X)) # Função de ativação de camada de entrada para a camada interna 1 
        X = F.relu(self.linear2(X)) # Função de ativação de camada interna 1 para a camada interna 2
        X = F.relu(self.linear3(X)) # Função de ativação de camada interna 2 para a camada da saída, nesse caso f(x) = x
        return F.log_softmax(X, dim=1) #Dados utilizados para calcular a perda

def treino(modelo, trainloader, device):
    
    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5) # Define a política de atualização dos pesos e da bias
    inicio = time() #Timer para sabermos quanto tempo levou o treino
    
    criterio = nn.NLLLoss() # Definindo o critério para calcular a perda
    EPOCHS = 10 # Número de epochs que o algoritmo rodará
    modelo.train() # treinando o modelo
    
    for epoch in range(EPOCHS):
        perda_acumulada = 0 #Inicialização da perda acumulada

        for imagens, etiquetas in trainloader: 
            
            imagens = imagens.view(imagens.shape[0], -1) #Convertendo as imagens para "vetores" de 28*28 casas da figura possíveis com a
            otimizador.zero_grad() #Zerando os gradientes por conta do ciclo anterior
            
            output = modelo(imagens.to(device)) #Colocando os dados no modelo
            perda_instantanea = criterio(output, etiquetas.to(device)) #Calculando a perda da epoch em questão
            
            perda_instantanea.backward()
            
            otimizador.step()
            
            perda_acumulada += perda_instantanea.item()#Atualização da perda acumulada
        else:
            print("Epoch {} - Perda resultante: {}".format(epoch+1, perda_acumulada/len(trainloader)))
    print("\nTempo de treino (em minutos) =",(time()-inicio)/60)

def validacao(modelo, valloader, device):
    conta_corretas, conta_todas = 0, 0
    for imagens, etiquetas in valloader:
        for i in range(len(etiquetas)):
            img = imagens[i].view(1, 784)
            #Desativar o autograd para caelerar a validação. Grafos computacionais dinâmicos tme um custo alto de processamento
            with torch.no_grad():
                logps = modelo(img.to(device))
        
        ps = torch.exp(logps) #Converte output para escala normal(lembranod que é um tensor)   
        probab = list(ps.cpi().numpy()[0])
        etiqueta_pred = probab.index(max(probab)) #converte o tensor em um número, no caso, o número que o modelo previu como correto
        etiqueta_certa = etiquetas.numpy()[i]
        if (etiqueta_certa == etiqueta_pred): #Compara a previsão com o valor correto
            conta_corretas += 1
        conta_todas += 1
        
    print("Total de imagens testadas = ", conta_todas)
    print("\nPrecisão do modelo = {}%".format(conta_corretas*100/conta_todas))  
            
#Baixar as imagens
transform = transforms.ToTensor()

trainset = datasets.MNIST('./MNIST_data/',download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

valset = datasets.MNIST('./MNIST_data/',download=True, train=False, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

for imagens, etiquetas in trainloader: #caso o imagens, etiquetas = dataiter.next() não funcione, use esse 'For'
    plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
    #plt.show()
    pass

#Dimensões
print(imagens[0].shape)
print(etiquetas[0].shape)

#Treinar o Modelo
modelo = Modelo()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(modelo.to(device))