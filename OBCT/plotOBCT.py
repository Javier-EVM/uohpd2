import plotly
import plotly.graph_objects as go
#import kaleido
#https://plotly.com/python/tree-plots/

#Define las coordenadas (x, y) para cada nodo del árbol
def plotOBCT(b,w,alpha,d):
    node_positions = {
        1: (0, 0),
        2: (-1, -1),
        3: (1, -1),
        4: (-1.5, -2),
        5: (-0.5, -2),
        6: (0.5, -2),
        7: (1.5, -2),
    }


    #Define las aristas de un árbol binario
    edges = [
        (1, 2),
        (1, 3),
        (2, 4),
        (2, 5),
        (3, 6),
        (3, 7)
    ]

    #Obtén las coordenadas x, y de los nodos y aristas
    Xn = [pos[0] for pos in node_positions.values()]
    Yn = [pos[1] for pos in node_positions.values()]

    Xe = []
    Ye = []
    for e in edges:
        Xe += [node_positions[e[0]][0],node_positions[e[1]][0],None]
        Ye += [node_positions[e[0]][1],node_positions[e[1]][1],None]

    #Etiquetas de los nodos
    labels = list(node_positions.keys())


    #print(Xn)
    #print(Yn)
    #print(Xe)
    #print(Ye)

    #Crea el gráfico
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=Xe,
        y=Ye,
        mode='lines',
        line=dict(color='rgb(30, 30, 30)', width=1),
        hoverinfo='none',
        opacity=0.2
    ))

    fig.add_trace(go.Scatter(
        x=Xn,
        y=Yn,
        mode='markers',
        name='Nodes',
        marker=dict(
            symbol='square-dot',
            size=80,
            color='#1d25cc',
            line=dict(color='rgb(150, 150, 250)', width=1),
        ),
        text=labels,
        hoverinfo='text',
        opacity=0.1
    ))

    # Agrega etiquetas como anotaciones de texto Gracias chatgpt!
    #b = {(1, 0): -0.0, (1, 1): 0.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): -0.0, (1, 5): 1.0, (1, 6): -0.0, (1, 7): 0.0, (1, 8): 0.0, (1, 9): -0.0, (1, 10): -0.0, (1, 11): -0.0, (1, 12): -0.0, (1, 13): -0.0, (1, 14): -0.0, (2, 0): 0.0, (2, 1): 0.0, (2, 2): 0.0, (2, 3): 0.0, (2, 4): 0.0, (2, 5): 0.0, (2, 6): 0.0, (2, 7): 1.0, (2, 8): 0.0, (2, 9): 0.0, (2, 10): 0.0, (2, 11): 0.0, (2, 12): 0.0, (2, 13): 0.0, (2, 14): 0.0, (3, 0): 0.0, (3, 1): 0.0, (3, 2): -0.0, (3, 3): -0.0, (3, 4): -0.0, (3, 5): 1.0, (3, 6): -0.0, (3, 7): -0.0, (3, 8): 0.0, (3, 9): 0.0, (3, 10): 0.0, (3, 11): 0.0, (3, 12): 0.0, (3, 13): -0.0, (3, 14): -0.0}
    #w = {(1, 0): -0.0, (1, 1): -0.0, (2, 0): 0.0, (2, 1): 0.0, (3, 0): 0.0, (3, 1): 0.0, (4, 0): 0.0, (4, 1): 1.0, (5, 0): 0.0, (5, 1): 1.0, (6, 0): 0.0, (6, 1): 1.0, (7, 0): 1.0, (7, 1): 0.0}

    labels = [str(node) for node in node_positions.keys()]
    #print(labels)
    for key in b:
        if b[key] == 1.0:
            labels[key[0]-1] = f"x[{key[1]}] == 1?"

    for key in w:
        if w[key] == 1.0:
            labels[key[0]-1] = f"class = {key[1]}"



    for i, label in enumerate(labels):
        fig.add_annotation(
            x=Xn[i],
            y=Yn[i],
            text=label,
            showarrow=False,
            font=dict(color='black', size=12)
        )
    
    fig.update_layout(
    title=f"OBCT lambda {alpha} d {d}",
    xaxis=dict(showgrid=False, showline=False, showticklabels=False),
    yaxis=dict(showgrid=False, showline=False, showticklabels=False)
    )

    plotly.offline.plot(fig, filename=f'OBCT/OBCT_tree_{alpha}_{d}.html')
    fig.show()


    """
    def positions(d):
        node_number = 2**d -1
        pos = {i+1 : [0,0]  for i in range(node_number)}
        return pos

    def calculate_positions(d,pos):
        node_number = 2**d -1
        NUL = range(node_number)

        for i in NUL:
            if i+1 == 1:
                pos[i] = [0,0]

                

            if 1 <= i < 2**(d-1):
                A[i] = [[math.floor(i/2),i],[i,2*i],[i,2*i + 1],[i,n_Nodos+1]]

            else:
                A[i] = [[math.floor(i/2),i],[-1,-1],[-1,-1],[i,n_Nodos+1]]

    print(positions(3))
    """