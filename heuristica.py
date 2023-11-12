
from VarRounding import takeMax2
from VarRounding import takeMax3
from VarRounding import takeMax_cut1
from MaxFlowOBCT import MFOBCT
def setMax(x_train, y_train,d ,lambda_):
    t1, of , b , w , z,gap = MFOBCT(x_train, y_train ,d ,lambda_ , False, False, True)
    #bl,wl = takeMax(b,w)
    bl,wl = takeMax2(b,w,d)
    t2, of , b , w , z,gap = MFOBCT(x_train, y_train ,d ,lambda_ , bl, wl, False)
    return (b,w, t1+t2)



def setMax3(x_train, y_train,d ,lambda_):
    t1, of , b , w , z,gap = MFOBCT(x_train, y_train ,d ,lambda_ , False, False, True)
    #bl,wl = takeMax(b,w)
    bl,wl = takeMax3(b,w,d)
    t2, of , b , w , z,gap = MFOBCT(x_train, y_train ,d ,lambda_ , bl, wl, False)
    return (b,w, t1+t2)



def setMax4(x_train, y_train,d ,lambda_,N):
    t1, of , b , w , z,gap = MFOBCT(x_train, y_train ,d ,lambda_ , False, False, True) #1 Relajación
    #bl,wl = takeMax(b,w)
    bl = takeMax_cut1(b,w,d) #1 b_{n,f} a fijar
    T = t1 #tiempo
    for i in range(N-1): #importante
        t2, of , b , w , z, gap = MFOBCT(x_train, y_train ,d ,lambda_ , bl, False, True) #N Relajaciones 
        bl = takeMax_cut1(b,w,d) #N b_{n,f} fijos
        T = T +t2 
    t2, of , b , w , z, gap = MFOBCT(x_train, y_train ,d ,lambda_ , bl, False, False) # Resolución final MIP
    T = T +t2
    #Finalmente
    #Se resuelven N+1 relajaciones
    #donde para relajacion 1 se usa relajacion uno
    #para relajacion N se usa relajación N (dentro del for es la numero N-1)
    #una fijación de variables no se usa, pero se utiliza fuera del for.
    #Resolucion final del MIP usa la fijación N+1
    #N = 0 implica 1 variable fijada
    #N = 1 implica 2 variable fijada
    #...
    #N = k implica k+1 variables fijadas
    #FIX
    #se recorre el for con N-1
    #N = 1 implica 1 variable fijada
    #N = 2 implica 2 variable fijada
    #...
    #N = k implica k variables fijadas
    return (b,w, T,gap)
