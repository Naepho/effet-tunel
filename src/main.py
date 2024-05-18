#!/usr/bin/env python3

from constants import ctes
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.interpolate import CubicSpline

def main():
    case = 6

    if case == 5:

        E_list = np.arange(0.1 * 1.6 * 10**(-19), 1.05 * 1.6 * 10**(-19), 0.05 * 10**(-19))
        a_list = np.arange(10**(-9), 10.5 * 10**(-9), 0.5 * 10**(-9))
        V_0_list = np.arange(0.1 * 1.6 * 10**(-19), 1.05 * 1.6 * 10**(-19), 0.05 * 10**(-19))
        n = 10

        a_len = len(a_list)
        V_0_len = len(V_0_list)
        E_len = len(E_list)
        solutions = np.zeros((a_len, V_0_len, E_len))
        counter = 0
        counter_max = a_len * V_0_len * E_len
        for a_index in range(a_len):
            for V_0_index in range(V_0_len):
                for E_index in range(E_len):
                    T, t = solve_problem(n, a_list[a_index], V_0_list[V_0_index], E_list[E_index])
                    solutions[a_index, V_0_index, E_index] = T
                    counter += 1
                    if counter % int(counter_max / 10) == 0:
                        print("Advancement : " + str(np.round(counter/counter_max * 100)) + " %")

        ### Graphique de T en fonction de a

        fig, (ax1, ax2) = plt.subplots(1, 2)

        V_0_index_1 = 2
        E_index_1 = E_len - 2
        V_0_index_2 = V_0_len - 2
        E_index_2 = 2

        sol_a_1 = solutions[:, V_0_index_1, E_index_1]
        sol_a_2 = solutions[:, V_0_index_2, E_index_2]

        a_list_nm = mtnm(a_list)

        xs = np.arange(a_list_nm[0], a_list_nm[-1], a_list_nm[0] / 100)
        sol_a_interp_1 = CubicSpline(a_list_nm, sol_a_1)
        sol_a_interp_2 = CubicSpline(a_list_nm, sol_a_2)

        ax1.plot(a_list_nm, sol_a_1, 'r.', label="E > V_0")
        ax1.plot(xs, sol_a_interp_1(xs), 'gray', linewidth = 0.5, label="Interpolation")
        ax1.set_xlabel("a [nm]", fontsize = 15)
        ax1.set_ylabel("T(E)", fontsize = 15)
        ax1.set_yscale('log')
        ax1.set_title("T en fonction de a, pour V_0 = " + str(np.round(jtev(V_0_list[V_0_index_1]), 10)) + " [eV], E = " + str(np.round(jtev(E_list[E_index_1]), 10)) + " [eV] et n = " + str(n), fontsize = 16)
        ax1.legend(fontsize = 15)

        ax2.plot(a_list_nm, sol_a_2, 'r.', label="E < V_0")
        ax2.plot(xs, sol_a_interp_2(xs), 'gray', linewidth = 0.5, label="Interpolation")
        ax2.set_xlabel("a [nm]", fontsize = 15)
        ax2.set_ylabel("T(E)", fontsize = 15)
        ax2.set_yscale('log')
        ax2.set_title("T en fonction de a, pour V_0 = " + str(np.round(jtev(V_0_list[V_0_index_2]), 10)) + " [eV], E = " + str(np.round(jtev(E_list[E_index_2]), 10)) + " [eV] et n = " + str(n), fontsize = 16)
        ax2.legend(fontsize = 15)

        plt.show()

        ### Graphique de T en fonction de V_0

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # a_index = a_len - 2
        # a = a_list[a_index]
        # E_index = int(E_len / 2)
        # E = E_list[E_index]
        # sol_V_0 = solutions[a_index, :, E_index]

        # V_0_list_1 = []
        # sol_V_0_1 = []
        # V_0_list_2 = []
        # sol_V_0_2 = []
        # for i in range(V_0_len):
        #     if E > V_0_list[i]:
        #         V_0_list_1.append(V_0_list[i])
        #         sol_V_0_1.append(sol_V_0[i])
        #     else:
        #         V_0_list_2.append(V_0_list[i])
        #         sol_V_0_2.append(sol_V_0[i])

        # V_0_list_1 = jtev(np.array(V_0_list_1))
        # V_0_list_2 = jtev(np.array(V_0_list_2))

        # ax1.plot(V_0_list_1, sol_V_0_1, 'r.', label="E > V_0")
        # ax2.plot(V_0_list_2, sol_V_0_2, 'r.', label="E < V_0")

        # V_0_interp_1 = CubicSpline(V_0_list_1, sol_V_0_1)
        # V_0_interp_2 = CubicSpline(V_0_list_2, sol_V_0_2)
        # xs1 = np.arange(V_0_list_1[0], V_0_list_1[-1], V_0_list_1[0] / 100)
        # xs2 = np.arange(V_0_list_2[0], V_0_list_2[-1], V_0_list_1[0] / 100)

        # ax1.plot(xs1, V_0_interp_1(xs1), 'gray', linewidth = 0.5, label="Interpolation par splines")
        # ax2.plot(xs2, V_0_interp_2(xs2), 'gray', linewidth = 0.5, label="Interpolation par splines")

        # ax1.set_xlabel("V_0 [eV]", fontsize = 14)
        # ax1.set_ylabel("T(E)", fontsize = 14)
        # ax1.set_yscale('log')
        # ax1.legend(fontsize = 14)

        # ax2.set_xlabel("V_0 [eV]", fontsize = 14)
        # ax2.set_ylabel("T(E)", fontsize = 14)
        # ax2.set_yscale('log')
        # fig.suptitle("T en fonction de V_0, pour a = " + str(np.round(mtnm(a), 7)) + " [nm], E = " + str(np.round(jtev(E), 10)) + " [eV] et n = " + str(n), fontsize = 15)
        # ax2.legend(fontsize = 14)

        # plt.show()

        ### Graphique de T en fonction de E

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # a_index = a_len - 2
        # a = a_list[a_index]
        # V_0_index = int(V_0_len / 2)
        # V_0 = V_0_list[V_0_index]
        # sol_E = solutions[a_index, V_0_index, :]

        # E_list_1 = []
        # sol_E_1 = []
        # E_list_2 = []
        # sol_E_2 = []
        # for i in range(E_len):
        #     if V_0 > E_list[i]:
        #         E_list_1.append(E_list[i])
        #         sol_E_1.append(sol_E[i])
        #     else:
        #         E_list_2.append(E_list[i])
        #         sol_E_2.append(sol_E[i])

        # E_list_1 = jtev(np.array(E_list_1))
        # E_list_2 = jtev(np.array(E_list_2))

        # ax1.plot(E_list_1, sol_E_1, 'r.', label="E < V_0")
        # ax2.plot(E_list_2, sol_E_2, 'r.', label="E > V_0")

        # E_interp_1 = CubicSpline(E_list_1, sol_E_1)
        # E_interp_2 = CubicSpline(E_list_2, sol_E_2)
        # xs1 = np.arange(E_list_1[0], E_list_1[-1], E_list_1[0] / 100)
        # xs2 = np.arange(E_list_2[0], E_list_2[-1], E_list_1[0] / 100)

        # ax1.plot(xs1, E_interp_1(xs1), 'gray', linewidth = 0.5, label="Interpolation par splines")
        # ax2.plot(xs2, E_interp_2(xs2), 'gray', linewidth = 0.5, label="Interpolation par splines")

        # ax1.set_xlabel("E [eV]", fontsize = 14)
        # ax1.set_ylabel("T(E)", fontsize = 14)
        # ax1.set_yscale('log')
        # ax1.legend(fontsize = 14)

        # ax2.set_xlabel("E [eV]", fontsize = 14)
        # ax2.set_ylabel("T(E)", fontsize = 14)
        # ax2.set_yscale('log')
        # fig.suptitle("T en fonction de E, pour a = " + str(np.round(mtnm(a), 7)) + " [nm], V_0 = " + str(np.round(jtev(V_0), 10)) + " [eV] et n = " + str(n), fontsize = 15)
        # ax2.legend(fontsize = 14)

        # plt.show()

    if case == 6:
        # a = 2.5 * 10**(-9)
        # V_0 = 0.5 * 1.6 * 10**(-19)
        # E = 0.25 * 1.6 * 10**(-19)

        # T101, t101 = solve_problem(101, a, V_0, E)

        # P = []
        # indices = []
        # for i in range(1, 101):
        #     P.append(Pn(i, a, V_0, E, T101, t101))
        #     indices.append(i)

        # print("Solution : " + str(indices[np.argmin(P)]))

        # # xs = np.arange(1, 100, 10**(-4))
        # # interp = CubicSpline(np.arange(1, 101), P)

        # plt.plot(np.arange(1, 101, 1), P, 'r.', label="P(n)")
        # plt.plot([1, 100], [np.min(P), np.min(P)], 'gray', linewidth = 0.5, label = "Minimum")
        # # plt.plot(xs, interp(xs), 'gray', linewidth = 0.5, label="Interpolation par splines")
        # plt.title("Évolution de P(n) avec n, pour a = " + str(mtnm(a)) + " [nm] , V_0 = " + str(jtev(V_0)) + " [eV], et E = " + str(jtev(E)) + " [eV]", fontsize = 16)
        # plt.xlabel("n", fontsize = 15)
        # plt.ylabel("P(n)", fontsize = 15)
        # plt.legend(fontsize = 15)
        # plt.show()

        E_list = np.arange(0.1 * 1.6 * 10**(-19), 1.05 * 1.6 * 10**(-19), 0.05 * 10**(-19))
        a_list = np.arange(10**(-9), 10.5 * 10**(-9), 0.5 * 10**(-9))
        V_0_list = np.arange(0.1 * 1.6 * 10**(-19), 1.05 * 1.6 * 10**(-19), 0.05 * 10**(-19))

        a_len = len(a_list)
        V_0_len = len(V_0_list)
        E_len = len(E_list)
        solutions = np.zeros((a_len, V_0_len, E_len))

        a_index_1 = int(a_len / 2)
        E_index_1 = E_len - 2
        a_index_2 = 2
        E_index_2 = 2
        V_0_index_1 = int(V_0_len / 2)
        V_0_index_2 = V_0_len - 2

        counter = 0
        counter_max = a_len * V_0_len * E_len
        for a_index in range(a_len):
            for V_0_index in range(V_0_len):
                for E_index in range(E_len):
                    if (V_0_index != V_0_index_1 or a_index != a_index_1) and (V_0_index != V_0_index_2 or a_index != a_index_2):
                        continue
                    n = optimize_pn(a_list[a_index], E_list[E_index], V_0_list[V_0_index])
                    solutions[a_index, V_0_index, E_index] = n
                    counter += 1
                    if counter % int(counter_max / 10) == 0:
                        print("Advancement : " + str(np.round(counter/counter_max * 100)) + " %")

        ### Graphique de opti en fonction de a

        # fig, (ax1, ax2) = plt.subplots(1, 2)

        # sol_a_1 = solutions[:, V_0_index_1, E_index_1]
        # sol_a_2 = solutions[:, V_0_index_2, E_index_2]

        # a_list_nm = mtnm(a_list)
        # V_0_list_ev = jtev(V_0_list)
        # E_list_ev = jtev(E_list)

        # ax1.plot(a_list_nm, sol_a_1, 'r.', label="E > V_0")
        # ax1.set_xlabel("a [nm]", fontsize = 15)
        # ax1.set_ylabel("n optimal", fontsize = 15)
        # ax1.set_title("n optimal en fonction de a, pour V_0 = " + str(np.round(jtev(V_0_list[V_0_index_1]), 10)) + " [eV], E = " + str(np.round(jtev(E_list[E_index_1]), 10)) + " [eV]", fontsize = 16)
        # ax1.legend(fontsize = 15)

        # ax2.plot(a_list_nm, sol_a_2, 'r.', label="E < V_0")
        # ax2.set_xlabel("a [nm]", fontsize = 15)
        # ax2.set_ylabel("n optimal", fontsize = 15)
        # ax2.set_title("n optimal en fonction de a, pour V_0 = " + str(np.round(jtev(V_0_list[V_0_index_2]), 10)) + " [eV], E = " + str(np.round(jtev(E_list[E_index_2]), 10)) + " [eV]", fontsize = 16)
        # ax2.legend(fontsize = 15)

        # plt.show()

        # Graphique en fonction de V_0

        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # a_index = a_index_1
        # E_index = E_index_1
        # a = a_list[a_index]
        # E = E_list[E_index]
        # sol_V_0 = solutions[a_index, :, E_index]

        # V_0_list_1 = []
        # sol_V_0_1 = []
        # V_0_list_2 = []
        # sol_V_0_2 = []
        # for i in range(V_0_len):
        #     if E > V_0_list[i]:
        #         V_0_list_1.append(V_0_list[i])
        #         sol_V_0_1.append(sol_V_0[i])
        #     else:
        #         V_0_list_2.append(V_0_list[i])
        #         sol_V_0_2.append(sol_V_0[i])

        # V_0_list_1 = jtev(np.array(V_0_list_1))
        # V_0_list_2 = jtev(np.array(V_0_list_2))

        # ax1.plot(V_0_list_1, sol_V_0_1, 'r.', label="E > V_0")
        # ax2.plot(V_0_list_2, sol_V_0_2, 'r.', label="E < V_0")

        # ax1.set_xlabel("V_0 [eV]", fontsize = 15)
        # ax1.set_ylabel("n optimal", fontsize = 15)
        # ax1.legend(fontsize = 15)

        # ax2.set_xlabel("V_0 [eV]", fontsize = 15)
        # ax2.set_ylabel("n optimal", fontsize = 15)
        # fig.suptitle("n optimal en fonction de V_0, pour a = " + str(np.round(mtnm(a), 7)) + " [nm], E = " + str(np.round(jtev(E), 10)) + " [eV]", fontsize = 16)
        # ax2.legend(fontsize = 15)

        # plt.show()

        # Graphique en fonction de E

        fig, (ax1, ax2) = plt.subplots(1, 2)
        a_index = a_index_1
        a = a_list[a_index]
        V_0_index = V_0_index_1
        V_0 = V_0_list[V_0_index]
        sol_E = solutions[a_index, V_0_index, :]

        E_list_1 = []
        sol_E_1 = []
        E_list_2 = []
        sol_E_2 = []
        for i in range(E_len):
            if V_0 > E_list[i]:
                E_list_1.append(E_list[i])
                sol_E_1.append(sol_E[i])
            else:
                E_list_2.append(E_list[i])
                sol_E_2.append(sol_E[i])

        E_list_1 = jtev(np.array(E_list_1))
        E_list_2 = jtev(np.array(E_list_2))

        ax1.plot(E_list_1, sol_E_1, 'r.', label="E < V_0")
        ax2.plot(E_list_2, sol_E_2, 'r.', label="E > V_0")

        ax1.set_xlabel("E [eV]", fontsize = 15)
        ax1.set_ylabel("n optimal", fontsize = 15)
        ax1.legend(fontsize = 15)

        ax2.set_xlabel("E [eV]", fontsize = 15)
        ax2.set_ylabel("n optimal", fontsize = 15)
        fig.suptitle("n optimal en fonction de E, pour a = " + str(np.round(mtnm(a), 7)) + " [nm], V_0 = " + str(np.round(jtev(V_0), 10)) + " [eV]", fontsize = 16)
        ax2.legend(fontsize = 15)

        plt.show()


    """
    # Tracer les rectangles et le triangle
    # Calcul des coordonnées des points du triangle

    x_values = [ctes["x_a"], ctes["x_a"] + ctes["a"]/2, ctes["x_a"] + ctes["a"]]
    y_values = [trig_barrier(ctes["x_a"], ctes["a"], ctes["V_0"], x_values[0]),
                trig_barrier(ctes["x_a"], ctes["a"], ctes["V_0"], x_values[1]),
                trig_barrier(ctes["x_a"], ctes["a"], ctes["V_0"], x_values[2])]

    # Affichage du triangle sur le plot
    plt.plot(x_values, y_values, 'r-', label='Triangle')
    plt.xlim(ctes["x_a"] * (1 - 0.3), (ctes["x_a"] + ctes["a"]) * (1 + 0.2))

    x_vals = np.linspace(ctes["x_a"], ctes["x_a"] + ctes["a"], ctes["n"] + 1)
    for i in range(1, len(rects_h) - 1):  # Correction : utilisez la longueur de rects_h comme référence
        plt.plot([x_vals[i-1], x_vals[i], x_vals[i], x_vals[i-1], x_vals[i-1]], [0, 0, rects_h[i], rects_h[i], 0], 'b-')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Triangle et rectangles')
    plt.legend()
    plt.grid(True)
    plt.show()
    """

def optimize_pn(a, E, V_0):
    T101, t101 = solve_problem(101, a, V_0, E)

    P = []
    indices = []
    for i in range(1, 101):
        P.append(Pn(i, a, V_0, E, T101, t101))
        indices.append(i)

    return indices[np.argmin(P)]


def Pn(n, a, V_0, E, T101, t101):
    Tn, tn = solve_problem(n, a, V_0, E)
    epn = np.abs((Tn - T101) / T101)
    Dtn = tn / t101

    return 4 * epn + Dtn

def solve_problem(n, a, V_0, E):
    start_time = timer()
    n = int(np.round(n))
    counter = 0

    M_T = np.identity(2)
    f = lambda x: trig_barrier(ctes["x_a"], a, V_0, x)
    rects_h = np.zeros(n + 2)
    rects_l = np.zeros_like(rects_h)
    for i in range(n):
        rect_l, rect_h = rectangle(f, n, ctes["x_a"], ctes["x_a"] + a, i + 1)
        rects_h[i + 1] = rect_h
        rects_l[i + 1] = rect_l
    rects_h[0] = 0
    rects_h[-1] = 0

    x = ctes["x_a"]
    for i in range(n + 1):
        if E == rects_h[i]:
            rects_h[i] *= 1.0000001 # increment of an epsilon
        if E == rects_h[i + 1]:
            rects_h[i + 1] *= 1.0000001 # increment of an epsilon

        k1 = np.emath.sqrt(2 * ctes["m"] * (E - rects_h[i]) / ctes["hb"]**2)
        k2 = np.emath.sqrt(2 * ctes["m"] * (E - rects_h[i + 1]) / ctes["hb"]**2)

        M1 = matrix(k1, x)
        M2 = np.linalg.inv(matrix(k2, x))
        Mtot = np.matmul(M2, M1)

        M_T = np.matmul(Mtot, M_T)

        x += rects_l[i + 1]

    frac = T(M_T)

    return frac, timer() - start_time

def rectangle(f, n, x_a, x_b, r_n):
    """
    Gives width and height of rectangle based on necessary values
    f : function to approximate
    n : number of rectangles used in approximation
    x_a : start of function domain
    x_b : end of function domain
    r_n : number of desired rectangle
    """

    l = (x_b - x_a) / n
    h = f(x_a + (r_n - 1) * l + (l / 2))

    return l, h

def line(a, b, x):
    # Just the function for a line
    return a * x + b

def trig_barrier(x_a, a, V_0, x):
    slope = V_0 / (a / 2)
    if x <= x_a or x >= x_a + a:
        return 0
    if x <= x_a + a/2:
        return line(slope, 0, x - x_a)
    if x <= x_a + a:
        return line(-1 * slope, V_0, x - x_a - a/2)

def matrix(k, x):
    M = np.array([[ np.exp(1j * k * x) , np.exp(-1j * k * x)], \
                   [ 1j * k * np.exp(1j * k * x) , -1j * k * np.exp(-1j * k * x) ]])

    return M

def T(M_T):
    return np.abs(M_T[0, 0] - ((M_T[0, 1] * M_T[1, 0]) / M_T[1,1]))**2

def mtnm(m):
    return m / 10**(-9)

def jtev(j):
    return j / (1.6 * 10**(-19))

if __name__=="__main__":
    main()
