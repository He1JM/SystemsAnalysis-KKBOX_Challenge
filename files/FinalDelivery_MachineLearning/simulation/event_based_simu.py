import numpy as np
import random 
import matplotlib.pyplot as plt


#Parametros para jugar
Tamaño_grid = 50 # Esto lo podemos cambiar
Iterations = 300 # suele ser 500 el mejor
#Probabilidades tambien para jugar 
Cancion_Repetir = 0.10
Cancion_Escuchar= 0.14
inflencia_prob= 0.35

no_influencia_prob= 0.01

influenciados_vecinos = 3 # debe haber vecinos con la influencia para que tenga sentido 



rejilla = np.zeros((Tamaño_grid,Tamaño_grid),dtype=int) # matriz n*n de tipo entero usaremos 1/0 

def count_neighbours(grid, row, col):
    grid = ()
    count = 0
    for pos in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1), (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)):
        if pos == 1:
            count += 1
    return count





historia = []

for t in range(Iterations):
    evento_paso = random.randint(1,8)
    for _ in range(evento_paso):
        x,y = random.randrange(Tamaño_grid),random.randrange(Tamaño_grid)
        r = random.random()
        
        if r < Cancion_Escuchar:
            #no se altero nada solo se escucho una cancion
            pass 
        elif r< Cancion_Escuchar+Cancion_Repetir: # Osea solo la escucho y la volvio a repetir deberia entrar en la rejilla
            rejilla[x,y]=1
            
        elif r < Cancion_Escuchar+Cancion_Repetir+inflencia_prob: # Validamos que escucho la cancion la repitio y se dio por su vecinos
            n = count_neighbours(rejilla,x,y) #falta un caos para la rejilla
            if n >= influenciados_vecinos and random.random() < inflencia_prob: # validamos que se puedan influenciar si no seria xD
                rejilla[x,y]=1 # pues si se influencio pues queda como 1 osea repitio cancion
        
        
        #Validamos la no influencia para que tenga sentido
        if random.random() < no_influencia_prob:
            rejilla[x,y] = 1- rejilla[x,y] # la dejamos como 0 por si no repite cancion 
            
        
        #Guardamos para poder imprimir
        historia.append(np.sum(rejilla))
        
        



plt.plot(historia)
plt.title("Simulacion event automata cellular")
plt.xlabel("Iteracion")
plt.ylabel("Celulas activas")
plt.show()

plt.imshow(rejilla, interpolation ="nearest") # para visualizar vecinos
plt.title("Rejilla de influencia")
plt.show()