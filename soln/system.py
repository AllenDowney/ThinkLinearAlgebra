#!/usr/bin/env python

# Standard library imports
from IPython.display import display, Math

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sympy as sp

# Local imports
from utils import decorate


def make_constant(G, name, **kwargs):
    symbol = sp.Symbol(name)
    G.graph['constants'].append(symbol)
    return symbol


def make_unknown(G, name, **kwargs):
    symbol = sp.Symbol(name)
    G.graph['unknowns'].append(symbol)
    return symbol

def add_symbols(G):
    # add currents to edges
    for u, v, data in G.edges(data=True):
        name = data['name']
        data['resistance'] = make_constant(G, name)
        data['current'] = make_unknown(G, f"I_{name}")

    # Assign voltages to nodes
    for node, data in G.nodes(data=True):
        if node in G.graph['fixed']:
            data['voltage'] = make_constant(G, f"V_{node}")
        else:
            data['voltage'] = make_unknown(G, f"V_{node}")


def make_voltage_divider():
    G = make_graph()
    G.graph['fixed'] = ['in', 'gnd']
    
    G.add_edge('in', 'out', component='R', name='R1')
    G.add_edge('out', 'gnd', component='R', name='R2')

    add_symbols(G)
    return G


def make_r2r_ladder_2bit():
    G = nx.DiGraph()
    G.graph['constants'] = []
    G.graph['unknowns'] = []
    G.graph['fixed'] = ['D1', 'D0', 'gnd']  # digital inputs and ground

    # Add R connections from digital inputs
    G.add_edge('D1', 'n1', component='R', name='R1')
    G.add_edge('D0', 'n2', component='R', name='R2')

    # Add 2R ladder
    G.add_edge('n1', 'n2', component='R', name='R3')   # from n1 to n2
    G.add_edge('n2', 'out', component='R', name='R4')  # from n2 to out
    G.add_edge('out', 'gnd', component='R', name='R5') # termination to GND

    # Attach symbols for resistors and currents
    for u, v, data in G.edges(data=True):
        name = data['name']
        if data['component'] == 'R':
            data['resistance'] = make_constant(G, name)
            data['current'] = make_unknown(G, f"I_{name}")

    # Assign voltages to all nodes
    for node, data in G.nodes(data=True):
        if node in G.graph['fixed']:
            data['voltage'] = make_constant(G, f"V_{node}")
        else:
            data['voltage'] = make_unknown(G, f"V_{node}")

    return G



def draw_graph(G):
    G.graph['graph'] = {'rankdir': 'LR'}
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_labels(G, pos)

    edge_labels = {(u, v): str(d['resistance']) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis('off')
    plt.tight_layout()
    


def make_ohm_equations(G):
    ohm_eqs = []
    
    for u, v, data in G.edges(data=True):
        if data['component'] == 'R':
            V_from = G.nodes[u]['voltage']
            V_to = G.nodes[v]['voltage']
            R = data['resistance']
            I = data['current']
            
            eq = sp.Eq(V_from - V_to, I * R)
            ohm_eqs.append(eq)
            
    return ohm_eqs


def make_kcl_equations(G):
    kcl_eqs = []
    
    for node in G.nodes:
        if node in G.graph['fixed']:
            continue

        expr_out = sum(data['current'] for _, _, data in G.out_edges(node, data=True))
        expr_in = sum(data['current'] for _, _, data in G.in_edges(node, data=True))
        eq = sp.Eq(expr_in, expr_out)
        
        kcl_eqs.append(eq)
        
    return kcl_eqs


def solve_circuit(G):
    ohm_eqs = make_ohm_equations(G)
    kcl_eqs = make_kcl_equations(G)
    
    print("Ohm's Law Equations:")
    for eqn in ohm_eqs:
        display(eqn)
    
    print("\nKCL Equations:")
    for eqn in kcl_eqs:
        display(eqn)
    
    eqs = ohm_eqs + kcl_eqs
    A, b = sp.linear_eq_to_matrix(eqs, G.graph['unknowns'])
    
    solution_vector = A.LUsolve(b)
    solution_vector = solution_vector.simplify()
    
    return solution_vector


if __name__ == "__main__":
    G = make_r2r_ladder_2bit()
    solution = solve_circuit(G)
    print(solution)

        