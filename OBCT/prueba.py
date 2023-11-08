import plotly.graph_objects as go

# Define las coordenadas (x, y) para cada nodo del árbol
node_positions = {
    'A': (0, 0),
    'B': (-1, -1),
    'C': (1, -1),
    'D': (-2, -2),
    'E': (0, -2),
}

# Define las aristas de un árbol binario
edges = [
    ('A', 'B'),
    ('A', 'C'),
    ('B', 'D'),
    ('B', 'E'),
]

# Obtén las coordenadas x, y de los nodos y aristas
Xn = [pos[0] for pos in node_positions.values()]
Yn = [pos[1] for pos in node_positions.values()]
Xe = [node_positions[src][0] for src, dst in edges] + [node_positions[dst][0] for src, dst in edges] + [None] * len(edges)
Ye = [node_positions[src][1] for src, dst in edges] + [node_positions[dst][1] for src, dst in edges] + [None] * len(edges)

# Etiquetas de los nodos
labels = list(node_positions.keys())

#https://stackoverflow.com/questions/74718871/plotly-plotting-traces-between-points
print(Xe)
print(Xe)
print(Xe)
print(Xe)
# Crea el gráfico
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=Xe,
    y=Ye,
    mode='lines',
    line=dict(color='rgb(210, 210, 210)', width=1),
    hoverinfo='none'
))
fig.add_trace(go.Scatter(
    x=Xn,
    y=Yn,
    mode='markers',
    name='Nodes',
    marker=dict(
        symbol='circle-dot',
        size=18,
        color='#6175c1',
        line=dict(color='rgb(50, 50, 50)', width=1),
    ),
    text=labels,
    hoverinfo='text',
    opacity=0.8
))

fig.show()