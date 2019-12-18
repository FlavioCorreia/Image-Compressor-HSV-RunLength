import numpy as np
from sys import getsizeof
import cv2
import matplotlib.pyplot as plt
import math

def converteRGBHSV(r,g,b):
    r /= 255
    g /= 255
    b /= 255

    menor = min(r, g, b)
    maior = max(r, g, b)    
    dif = maior-menor
    if maior == menor:
        h = 0
    elif maior == r and g >= b: #h
        h = 60*(g-b)/dif
    elif maior == r and g < b:
        h = 60*(g-b)/dif+360
    elif maior == g:
        h = 60*(b-r)/dif+120
    else:
        h = 60*(r-g)/dif+240
    
    if maior == 0: #s
        s = 0
    else: 
        s = (dif / maior)
  
    v = maior
    
    return [h, math.ceil(s*100), math.ceil(v*100)] # h°, s, v (s e v arredondado p/ cima)

def converteHSVRGB(h,s,v):
    if h > 1:
        h /= 360
    if s > 1:
        s /= 100
    if v > 1:
        v /= 100

    if v == 0:
        return 0,0,0
    
    piso = (h * 6)//1
    f = h * 6 - piso;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);

    if piso%6 == 0:
        r,g,b = v,t,p
    elif piso%6 == 1:
        r,g,b = q,v,p
    elif piso%6 == 2:
        r,g,b = p,v,t
    elif piso%6 == 3:
        r,g,b = p,q,v
    elif piso%6 == 4:
        r,g,b = t,p,v
    elif piso%6 == 5:
        r,g,b = v,p,q

    r,g,b = math.ceil(r*255), math.ceil(g*255), math.ceil(b*255)
    return [r,g,b]


def runLenCompactar(img, nomeCompactado, iIni = 426, tamA = 18): #427 / 214 COMPACT RUN L
    #print(img[427,0]);print(img[427,1]);print(img.shape)
    if tamA+iIni > img.shape[0]:
        tamA = img.shape[0] - iIni #GARANTIR Q N VAI ACESSAR FORA
        
    tamL = img.shape[1]
    
    b,g,r = img[iIni,0]
    lista = [[0,r,g,b]]
    for i in range(iIni, iIni+tamA):
        for j in range(tamL):
            b,g,r = img[i,j]
            if lista[-1][1] == r and lista[-1][2] == g and lista[-1][3] == b:
                lista[-1][0] += 1
            else:
                lista += [[1, r, g, b]]

    np.save(nomeCompactado+"RL.npy", lista)
    return lista


def runLenDescompactar(img, nomeNpy, nomeSaida, iIni = 426, tamA = 18): #DESCOMPACTA RUN L
    lista = np.load(nomeNpy)
    idLista = 0;
    tamL = img.shape[1]
    for i in range(iIni, iIni+tamA):
        for j in range(tamL):
            if lista[idLista][0] == 0:
                idLista += 1
                
            lista[idLista][0] -= 1
            img[i,j] = [lista[idLista][3], lista[idLista][2], lista[idLista][1]] #BGR
               
    cv2.imwrite(nomeSaida, img)
    
def hsvCompactar(img, nomeCompactado): #COMPACTA HSV
    def pegaValores(i,j):
        p1 = converteRGBHSV(img[i,j,0], img[i,j,1], img[i,j,2])
        p2 = converteRGBHSV(img[i+1,j,0], img[i+1,j,1], img[i+1,j,2])
        p3 = converteRGBHSV(img[i,j+1,0], img[i,j+1,1], img[i,j+1,2])
        p4 = converteRGBHSV(img[i+1,j+1,0], img[i+1,j+1,1], img[i+1,j+1,2])

        sM = (p1[1]+p2[1]+p3[1]+p4[1])/4
        vM = (p1[2]+p2[2]+p3[2]+p4[2])/4
        return (p1[0], p2[0], p3[0], p4[0], sM, vM)

    tamA = int(img.shape[0]/2)
    tamL = int(img.shape[1]/2)

    imgF = np.zeros((tamA, tamL, 6)).astype(np.uint8)# NOVA IMG   
    for i in range(tamA):
        for j in range(tamL):
            #if i < 213 or i > 222: # (426-444 é o texto)
            imgF[i,j] = pegaValores(2*i,2*j)

    if img.shape[0] > 425: #USAR O RUN LENGHT
        runLenCompactar(img, nomeCompactado)

    np.save(nomeCompactado+".npy", imgF) #SEMPRE TERÁ O HSV

def hsvDescompactar(nomeCompactado, nome): #DESCOMPACTA HSV     
    if nome[-4:] != ".bmp":
        nome += ".bmp"
        
    matriz = np.load(nomeCompactado+".npy")
    #imgF = np.ones((1, 1, 3)).astype(np.uint8)
    
    if len(matriz.shape) == 3:
        if matriz.shape[2] == 6: #HSV (H1, H2, H3, H4, SM, VM)
            tamA = matriz.shape[0]
            tamL = matriz.shape[1]
            imgF = np.ones((tamA*2, tamL*2, 3)).astype(np.uint8)

            for i in range(tamA):
                for j in range(tamL):
                    #if i < 213 or i > 222: #FORA DO ESCOPO DO RL
                    sM = matriz[i,j,4]
                    vM = matriz[i,j,5]
                        
                    p1 = converteHSVRGB(matriz[i,j,0], sM, vM)
                    p2 = converteHSVRGB(matriz[i,j,1], sM, vM)
                    p3 = converteHSVRGB(matriz[i,j,2], sM, vM)
                    p4 = converteHSVRGB(matriz[i,j,3], sM, vM)
                            
                    imgF[2*i,2*j] = p1
                    imgF[2*i+1,2*j] = p2
                    imgF[2*i,2*j+1] = p3
                    imgF[2*i+1,2*j+1] = p4

    try:
        qqCoisa = np.load(nomeCompactado+"RL.npy") #ver se existe o rl para aquela compactação
        runLenDescompactar(imgF, nomeCompactado+"RL.npy", nome)
        
    except:
        cv2.imwrite(nome, imgF)
                    
#img1 = cv2.imread("benchmark.bmp", cv2.COLOR_BGR2RGB) #descompactada0
#img1 = cv2.imread("tigre.bmp", cv2.COLOR_BGR2RGB)
#img1 = cv2.imread("kindred.bmp", cv2.COLOR_BGR2RGB)


#hsvCompactar(img, "aviaoComp")                                    #<------------------------

#hsvDescompactar("aviaoComp", "aviaoDesc")                          #<------------------------

#----------------------------------------INTERFACE
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox

d = {'acao': 1, 'db': 0} # 'acao' 1 -> compactar / 'acao' 0 -> descompactar

def janela():
    window = Tk() 
    window.title("COMPRESSOR DE IMAGENS PYTHON")
    window.minsize(width=700, height=350)
    window.maxsize(width=700, height=350)
    window.resizable(0,0)

    lblINUTIL = Label(window, text="")     
    lblINUTIL.grid(column=0, row=0, padx=10, pady = 20)
    
    lblIE = Label(window, text="Nome da Imagem de Entrada:")     
    lblIE.grid(column=0, row=1, padx=10, pady = 20)
    entrada1 = Entry(window, width=8)
    entrada1.grid(column=1, row=1, pady = 20, ipady = 8, ipadx = 200)
    

    lblIS = Label(window, text="Nome do Arquivo Compactado:")     
    lblIS.grid(column=0, row=3, padx=10, pady = 0)
    entrada2 = Entry(window, width=8)
    entrada2.grid(column=1, row=3, pady = 0, ipady = 8, ipadx = 200)

    def mudarFuncionalidade():
        global d
        d['acao'] = (d['acao']+1)%2
        if d['acao'] == 1:
            btnA["text"]="Compactar Imagem"
            lblIE["text"]="Nome da Imagem de Entrada:"
            lblIS["text"]="Nome do Arquivo Compactado:"
            
        else:
            btnA["text"]="Descompactar Imagem"
            lblIE["text"]="Nome do Arquivo Compactado:"
            lblIS["text"]="Nome da Imagem de Saída:"

    btnA = Button(window, text="Compactar Imagem", command=mudarFuncionalidade)
    btnA.grid(column=1, row=4, pady = 25)
    

    def chamaErro(titulo="Erro", msg=""):
        messagebox.showerror(titulo, msg)
    
    def executarAcao():
        if d['acao'] == 1: #COMPACTAR
            img = cv2.imread(entrada1.get(), cv2.COLOR_BGR2RGB)
            if str(type(img)) != "<class 'NoneType'>":
                if len(entrada1.get()) > 5 and entrada1.get()[-3:] in ["bmp", "BMP"]:
                    if entrada2.get()[-3:] != "npy":
                        
                        hsvCompactar(img, entrada2.get())
                        messagebox.showinfo('','TERMINOU COMPACTAÇÃO COM SUCESSO')
                        
                    else:
                        chamaErro("Erro", "Arquivo de saída inválido, não informe formato para ele")
                else:
                    chamaErro("Erro", "Imagem de entrada inválida, use somente imagens .bmp")
            else:
                chamaErro("Erro", "Imagem de entrada não existe")           

        else: #DESCOMPACTAR
            if entrada1.get()[-3:] not in ["npy", "NPY"]:
                imagemDeSaida = entrada2.get()
                if imagemDeSaida[-4:] in [".bmp", ".BMP"]:
                    imagemDeSaida = imagemDeSaida[:-4]                    

                hsvDescompactar(entrada1.get(), imagemDeSaida)
                messagebox.showinfo('','TERMINOU DESCOMPACTAÇÃO COM SUCESSO')
                
            else:
                chamaErro("Erro", "Arquivo de entrada inválido, não informe formato para ele")  
        
    btnOk = Button(window, text="Executar", command=executarAcao) 
    btnOk.grid(column=1, row=6, pady = 25)

    window.mainloop()

janela()
































































