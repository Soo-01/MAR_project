import sympy as sp

t1, t2, t3, t4, t5, t6 = sp.symbols("theta1 theta2 theta3 theta4 theta5 theta6")

def Tx(x):
    return sp.Matrix([
        [1, 0, 0, x],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

def Ty(y):
    return sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

def Tz(z):
    return sp.Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])

def Trans(x, y, z):
    return sp.Matrix([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1],
    ])

def Rx(theta):
    c = sp.cos(theta)
    s = sp.sin(theta)
    return sp.Matrix([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ])

def Ry(theta):
    c = sp.cos(theta)
    s = sp.sin(theta)
    return sp.Matrix([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ])

def Rz(theta):
    c = sp.cos(theta)
    s = sp.sin(theta)
    return sp.Matrix([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

A1 = Trans(0.03350, 0, 0.07105) * Rz(t1)
A2 = Trans(0, 0, 0.0565) * Ry(t2)
A3 = Trans(0.024, 0.00005, 0.1445) * Ry(t3)
A4 = Trans(0.02535, 0, 0.18165) * Rz(t4)
A5 = Trans(0, 0, 0.06010) * Rx(t5)
A6 = Trans(0.00005, -0.016, 0.0875) * Rz(t6)

T1 = sp.simplify(A1)
T2 = sp.simplify(A1 * A2)
T3 = sp.simplify(A1 * A2 * A3)
T4 = sp.simplify(A1 * A2 * A3 * A4)
T5 = sp.simplify(A1 * A2 * A3 * A4 * A5)
T6 = sp.simplify(A1 * A2 * A3 * A4 * A5 * A6)

sp.print_latex(T6)