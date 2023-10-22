import math


def tree(d):
    #numero de nodos del arbol, sin contar s y t
    n_Nodos = 2**d -1

    s = 0
    t = 2**d

    N = range(1, 2**(d-1))
    #print(N)
    L = range(2**(d-1) , n_Nodos+1)
    NUL = range(1,n_Nodos+1)
    s_NUL = range(0,n_Nodos+1)
    NUL_t = range(1,n_Nodos+2) #revisar


    A = {}

    for i in NUL:
        if i == 0:
            A[i] = [[0,1],[-1,-1],[-1,-1],[1,n_Nodos+1]]

        if 1 <= i < 2**(d-1):
            A[i] = [[math.floor(i/2),i],[i,2*i],[i,2*i + 1],[i,n_Nodos+1]]

        else:
            A[i] = [[math.floor(i/2),i],[-1,-1],[-1,-1],[i,n_Nodos+1]]

    return(s, t, N, L, NUL, s_NUL, NUL_t, A)