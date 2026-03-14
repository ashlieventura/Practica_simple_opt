import numpy as np
import re
from typing import Optional

import re
import numpy as np

def read_dades(num: int, prob: int, fitxer: str = "OPT25-26_Datos práctica 1.txt"):
    """Llegeix dades d'un fitxer i n'extreu la informació rellevant.

    Args:
        num (int): Número per identificar les dades de l'estudiant.
        prob (int): Número per identificar el problema.
        fitxer (str): Ruta del fitxer a llegir.

    Returns:
        cost (np.ndarray): Coeficients de cost de la funció objectiu.
        A (np.ndarray): Matriu de coeficients de les restriccions.
        b (np.ndarray): Valors de les restriccions.
        z (float | None): Valor de la funció objectiu, o None si no es troba.
        v (list | None): Llista d'índexs de la base, o None si no es troba.
    """
    def extract_ints(line: str) -> list[int]:
        return list(map(int, re.findall(r'-?\d+', line)))

    def is_number_line(line: str) -> bool:
        return bool(re.match(r'^[\d\- ]+$', line.strip())) and line.strip() != ""

    with open(fitxer, "r") as f:
        lines = f.readlines()

    # --- Cerca la capçalera de la secció ---
    tag_num  = f"datos{num}".lower()
    tag_prob = f"problemaPL{prob}".lower()
    start = next(
        (i for i, l in enumerate(lines)
         if tag_num in l.replace(" ", "").lower()
         and tag_prob in l.replace(" ", "").lower()),
        None
    )

    # --- Vector de costos ---
    cost = []
    while idx < len(lines) and "A=" not in lines[idx]:
        if is_number_line(lines[idx]):
            cost += extract_ints(lines[idx])
        idx += 1

    # --- Matriu A (gestiona blocs de múltiples columnes) ---
    A = []
    primer_bloc = True
    idx += 1  # salta la línia "A="
    while idx < len(lines) and "b=" not in lines[idx]:
        line = lines[idx]
        if "Column" in line:
            if primer_bloc:
                primer_bloc = False
                idx += 1
            else:
                # Bloc següent: afegeix columnes a les files existents
                idx += 1
                for i in range(len(A)):
                    if idx < len(lines):
                        A[i] += extract_ints(lines[idx])
                        idx += 1
            continue
        if is_number_line(line):
            A.append(extract_ints(line))
        idx += 1

    # --- Vector b ---
    idx += 1  # salta la línia "b="
    b = extract_ints(lines[idx])
    idx += 1

    # --- Valor òptim z* i base v ---
    z, v = None, None
    while idx < len(lines):
        line = lines[idx]
        if "z*=" in line and "---" not in line:
            idx += 1
            z = float(lines[idx].strip())
            idx += 3  # salta el valor de z i 2 línies fins arribar a v
            v = extract_ints(lines[idx])
            break
        idx += 1

    return np.array(cost), np.array(A), np.array(b), z, v





"""

def simplex(cost: np.array, A: np.array, b: np.array, inversa: Optional[np.array] = None, fase1: bool = None):
    Performs the simplex algorithm.

    Args:
    - cost (numpy.ndarray): Cost coefficients for the linear program objective function.
    - A (numpy.ndarray): Coefficients of the constraints matrix.
    - b (numpy.ndarray): Values of the constraints.
    - inversa (numpy.ndarray, optional): Inverse matrix. Defaults to None.
    - fase1 (bool, optional): Indicates if it's the first phase of the simplex algorithm. Defaults to None.

    Returns:
    - x (numpy.ndarray or None): Solution vector.
    - z (float or str or None): Objective function value. None if infactible, No acotat if unbounded.
    - basiques (list): Basic variables.
    - inversa (numpy.ndarray or None): Inverse matrix.
    - iteracio (int): Number of iterations.

    with open ("output.txt", "a") as doc:
        iteracio = 0
        m = len(b)
        n = len(A[0])
        if(n < m):
            return None, "Sistema incorrecte", [], None, None
        no_basiques = [a for a in range(n-m)]
        basiques = [a for a in range(n) if a not in no_basiques]
        basiques_noves = None
        if fase1 == None:
            nova_A = np.hstack((A, np.eye(m))) # horizontal stack
            nou_cost = np.array([0 for _ in range(n)] + [1 for _ in range(m)])
            x_f1, z_f1, basiques_noves, inv, iteracions = simplex(nou_cost, nova_A, b, np.eye(m), fase1 = True)
            inversa = inv
            iteracio = iteracions + 1
            if z_f1 in [None, "Infactible", "No acotat"] or np.round(z_f1,10) > 0:
                    doc.write("No hi ha solucio factible\n\n")
                    return None, "Infactible", [], None, iteracio
            if min(x_f1) == 0 and basiques_noves[np.argmin(x_f1)] >= n:
                # Cas en el que fase I acaba amb degeneració en una de les variables artificials.
                """
"""              Una alternativa és buscar qualsevol no bàsica amb cost reduït 0 i l'element de la
                qual en la fila de la variable artificial sigui diferent a 0.
                
                marxa = basiques_noves[np.argmin(x_f1)]
                basiques_per_iterar = [a for a in basiques if a not in basiques_noves]
                for idx in basiques_per_iterar:
                    try:
                        basiques_temp = basiques_noves.copy()
                        basiques_temp[np.argmin(x_f1)] = idx
                        B = A[:,basiques_temp]
                        inversa = np.linalg.inv(B)
                        basiques_noves[np.argmin(x_f1)] = idx
                        break
                    except:
                        continue
                else:
                    doc.write("SBF inicial contenia una variable artificial que no s'ha pogut treure.\n")
                    cost = np.append(cost, (np.zeros(m)))
                    z = np.dot(cost[basiques_noves], x_f1)
                    doc.write(f"x = {x_f1}\nz* = {z}\nbasiques = {basiques_noves}\n\n")
                    return x_f1, z, basiques_noves, None, iteracio
            basiques = basiques_noves
            no_basiques = [a for a in range(n) if a not in basiques]
            doc.write("SBF inicial trobada.\n\n")
            doc.write("Fase 2\n")
        else:
            doc.write("Fase 1\n")
        cost_b = cost[basiques]
        B_inv = inversa
        x = np.dot(B_inv, b)
        z = np.dot(cost_b, x)

        if m == n: # Si hi ha el mateix nombre de variables que restriccions, la SBF és la única
            x = np.dot(np.linalg.inv(A), b)
            z = np.dot(cost, x)
            return x, z, basiques, None, iteracio
        
        while True:
            degenerat = False
            B = A[:,basiques]
            A_n = A[:,no_basiques]
            B_inv = inversa
            cost_b = cost[basiques]
            cost_n = cost[no_basiques]
            if np.round(min(x),10) == 0:
                doc.write("Solucio degenerada\n")
                degenerat = True
                pass
            elif min(x) < 0:
                return x, None, None, None, iteracio
            
            # és optim?
            r = np.subtract(cost_n, np.dot(np.dot(cost_b, B_inv),A_n))
            if min(r) < 0:
                pass
            elif min(r) > 0:
                doc.write("Solucio optima trobada\n\n") if fase1 != True else doc.write("Fi Fase I\n\n")
                doc.write(f"x = {x}\nz* = {z}\nbasiques = {basiques}\nr = {r}\n\n") if fase1 != True else doc.write(f"basiques = {basiques}\n\n")
                doc.write(f"Nombre d'iteracions: {iteracio + 1}\n\n")
                return x, z, basiques, inversa, iteracio
            else:
                doc.write("Una de les solucions optimes trobada\n\n") if fase1 != True else doc.write("Fi Fase I\n\n")
                doc.write(f"x = {x}\nz* = {z}\nbasiques = {basiques}\nr = {r}\n\n") if fase1 != True else doc.write(f"basiques = {basiques}\n\n")
                doc.write(f"Nombre d'iteracions: {iteracio + 1}\n\n")
                return x, z, basiques, inversa, iteracio
            for e in range(len(r)):
                if r[e] < 0:
                    break
            entra = no_basiques[e]
            # direcció bàsica factible
            d_B = -np.dot(B_inv, A_n[:,e])

            if min(d_B) >= 0:
                doc.write("Optim no acotat (raig)\n\n")
                doc.write(f"basiques = {basiques}\ndB = {d_B}\n\n")
                doc.write(f"Nombre d'iteracions: {iteracio + 1}\n\n")
                return x, "No acotat", basiques, inversa, iteracio
            # longitud de pas
            theta, marxa = min([(np.divide((-x[i]),d_i), basiques[i]) for i, d_i in enumerate(d_B) if d_i < 0])

            if degenerat and theta == 0:
                
                Aquest condicional està per la propietat ii de la diapositiva 30.
                No vam acabar d'entendre exactament en quins casos es donava aquest
                problema
                
                if max([np.divide((-x[i]),d_i) for i, d_i in enumerate(d_B) if d_i < 0]) == 0:
                    # No hi ha cap theta > 0 (lo de la presentació?)
                    # return x, "Infactible", basiques, None
                    pass
                else:
                    pass
                    

            p = basiques.index(marxa)
            basiques[p] = entra
            no_basiques = [a for a in range(n) if a not in basiques] # Ordenades
            doc.write(f"q = {entra}, out = {marxa}, p = {p}, theta = {theta}, z = {z}\n")
            
            # actualitzacions
            transformacio = np.eye(m)
            transformacio[:,p] = [np.divide((-d_B[i]), d_B[p]) if i!=p else np.divide((-1),d_B[p]) for i in range(m)]
            inversa = np.dot(transformacio, B_inv)

            x += np.dot(theta,d_B)
            x[p] = theta

            z += np.dot(r[e],theta)
            iteracio += 1

for alumne in range(1, 67):
    for problema in range(1, 5):
        print(f"Alumne {alumne}, problema {problema}")
        c, A, b, z ,v = read_dades(alumne,problema)
        with open ("output.txt", "a") as doc:
            doc.write(f"Alumne {alumne}, problema {problema}\n")
        a = simplex(c, A, b, None)
        if z != None:
            result = f"{a[1]:.4f}" == f"{z:.4f}"
            print(result, f"{a[1]:.4f}")
        else:
            print(a[1])
        print(f"Iteracions: {a[4]}")


print("------------- \n Test \n --------------")
with open ("output.txt", "a") as doc:
    doc.write("------------------------------------------------------------------------------\n")
    doc.write("                                 Test\n")
    doc.write("------------------------------------------------------------------------------\n\n")
for alumne in range(67, 71):
    for problema in range(1, 5):
        print(f"Alumne {alumne}, problema {problema}")
        c, A, b, z ,v = read_dades(alumne,problema, fitxer="Datos_práctica_1_test.txt")
        with open ("output.txt", "a") as doc:
            doc.write(f"Alumne {alumne}, problema {problema}\n")
        a = simplex(c, A, b, None)
        print(a[1])
Per comprovar:

from scipy.optimize import linprog
r = linprog(c, A_eq = A, b_eq = b, method='highs')
if r['status']==0:
    if a[1] != None and a[1] != "?":
        print(f"{r['fun']:.4f}", f"{a[1]:.4f}")
    else:
        print("a was none and r was ", r['fun'])
elif r['status'] == 2:
    print("infactible")
    if a[1] == "Infactible":
        print(True)
    else:
        print(False)
elif r['status'] == 3:
    print("raig")
    if a[1] == "No acotat":
        print(True)
    else:
        print(False)
"""
