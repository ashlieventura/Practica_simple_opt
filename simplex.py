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



def fase_inicial(A: np.ndarray, b: np.ndarray):
    """Troba una SBF inicial mitjançant el mètode de la fase I (variables artificials).

    Construeix el problema auxiliar:
        min  sum(a_i)
        s.a. [A | I] * [x; a] = b,  x,a >= 0
    on a_i són m variables artificials amb cost 1 i la resta cost 0.
    Resol el problema auxiliar amb simplex_proces partint de la base artificial I.
    Si el mínim és 0 (dins tolerància), extreu la base factible original.

    Args:
        A (np.ndarray): Matriu de restriccions (m x n) del problema original.
        b (np.ndarray): Termes independents (m,). Han de ser >= 0.

    Returns:
        basiques (list[int]): Índexs de les variables bàsiques de la SBF.
        no_basiques (list[int]): Índexs de les variables no bàsiques.
        inversa (np.ndarray): Inversa de la base B de la SBF.
        factible (bool): True si s'ha trobat una SBF, False si el problema és infactible.
    """
    m, n = A.shape

    # Garantim b >= 0 (multipliquem files negatives per -1)
    A_aux = A.astype(float).copy()
    b_aux = b.astype(float).copy()
    for i in range(m):
        if b_aux[i] < 0:
            A_aux[i] *= -1
            b_aux[i] *= -1

    # Problema auxiliar: [A_aux | I_m], cost = [0,...,0, 1,...,1]
    A_fase1 = np.hstack((A_aux, np.eye(m)))
    cost_fase1 = np.array([0.0] * n + [1.0] * m)

    # Base inicial: les m variables artificials (índexs n, n+1, ..., n+m-1)
    basiques_f1 = list(range(n, n + m))
    inversa_f1 = np.eye(m)

    # Resol el problema auxiliar
    x_f1, z_f1, bas_f1, inv_f1, _ = simplex_proces(
        cost_fase1, A_fase1, b_aux, basiques_f1, inversa_f1
    )

    # Si z* > 0 (dins tolerància numèrica) → infactible
    if z_f1 == "No acotat" or np.round(float(z_f1), 10) > 0:
        return [], list(range(n)), None, False

    # Elimina les variables artificials de la base si hi han quedat amb valor 0
    basiques = list(bas_f1)
    inversa = inv_f1.copy()
    for pos, var in enumerate(basiques):
        if var >= n:
            # Variable artificial a la base amb valor 0 (degeneració)
            # Intentem substituir-la per qualsevol variable original no bàsica
            no_bas_orig = [j for j in range(n) if j not in basiques]
            substituida = False
            for j in no_bas_orig:
                col = inversa @ A_aux[:, j]
                if abs(col[pos]) > 1e-10:
                    # Aplica el pivot per treure la variable artificial
                    eta = np.eye(m)
                    eta[:, pos] = [-col[i] / col[pos] if i != pos else 1.0 / col[pos]
                                   for i in range(m)]
                    inversa = eta @ inversa
                    basiques[pos] = j
                    substituida = True
                    break
            # Si no s'ha pogut substituir, la columna és linealment dependent;
            # la variable artificial pot quedar (la fila és redundant).

    no_basiques = sorted([j for j in range(n) if j not in basiques])
    return basiques, no_basiques, inversa, True


def simplex_proces(cost: np.ndarray, A: np.ndarray, b: np.ndarray,
                   basiques: list, inversa: np.ndarray):
    """Executa les iteracions del simplex primal amb la regla de Bland.

    Args:
        cost (np.ndarray): Coeficients de cost.
        A (np.ndarray): Matriu de restriccions (m x n).
        b (np.ndarray): Termes independents.
        basiques (list): Índexs de les variables bàsiques inicials.
        inversa (np.ndarray): Inversa de la base B inicial.

    Returns:
        x (np.ndarray): Vector solució bàsica.
        z (float | str): Valor de z* o 'No acotat'.
        basiques (list): Variables bàsiques finals.
        inversa (np.ndarray | None): Inversa de la base final.
        iteracio (int): Nombre d'iteracions realitzades.
    """
    m, n = A.shape
    basiques = list(basiques)
    no_basiques = sorted([j for j in range(n) if j not in basiques])
    B_inv = inversa.copy()

    # Solució bàsica inicial i valor de la funció objectiu
    x = B_inv @ b
    z = float(cost[basiques] @ x)
    iteracio = 0

    while True:
        cost_b = cost[basiques]
        cost_n = cost[no_basiques]

        # Pas 2: costos reduïts  r' = C_N - C_B * B^{-1} * A_N
        r = cost_n - cost_b @ B_inv @ A[:, no_basiques]

        # Condició d'optimalitat: r' >= 0  → STOP
        candidats = [(no_basiques[e], e) for e in range(len(r)) if r[e] < 0]
        if not candidats:
            return x, z, basiques, B_inv, iteracio

        # Regla de Bland: variable d'entrada q = menor índex amb r_q < 0
        entra_idx, e = min(candidats, key=lambda t: t[0])

        # Pas 3: direcció bàsica factible  d_B = -B^{-1} * A_q
        d_B = -(B_inv @ A[:, entra_idx])

        # Pas 3.2: totes components >= 0 → PL no acotat → STOP
        if np.all(d_B >= 0):
            return x, "No acotat", basiques, B_inv, iteracio

        # Pas 4: longitud de pas màxima  θ* = min_{i | d_B(i)<0} -x_B(i)/d_B(i)
        # Desempat per Bland: menor índex de la variable bàsica de sortida
        ratios = [
            (-x[i] / d_B[i], basiques[i], i)
            for i in range(m) if d_B[i] < 0
        ]
        theta, _, p = min(ratios, key=lambda t: (t[0], t[1]))
        marxa = basiques[p]

        # Pas 5.1: actualitzar x_B i z
        x = x + theta * d_B
        x[p] = theta
        z = z + theta * r[e]

        # Pas 5.2: actualitzar conjunts B i N
        basiques[p] = entra_idx
        no_basiques = sorted([j for j in range(n) if j not in basiques])

        # Actualitzar la inversa per eta-factorització
        transformacio = np.eye(m)
        transformacio[:, p] = [
            -d_B[i] / d_B[p] if i != p else -1.0 / d_B[p]
            for i in range(m)
        ]
        B_inv = transformacio @ B_inv

        iteracio += 1


