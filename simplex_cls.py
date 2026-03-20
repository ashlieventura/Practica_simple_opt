import re
from typing import Optional

import numpy as np


class Simplex:
    """
    Implementación del simplex primal en dos fases (Fase I + Fase II).
    Utiliza la regla de Bland para la selección de variables entrantes y salientes.
    """

    def __init__(
        self,
        fitxer: Optional[str] = None,
        num: Optional[int] = None,
        prob: Optional[int] = None,
        cost: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        z_sol: Optional[float] = None,
        x_sol: Optional[list[int]] = None,
    ) -> None:
        self.fitxer = fitxer
        self.num = num
        self.prob = prob

        self.cost = cost
        self.A = A
        self.b = b
        self.z_sol = z_sol
        self.x_sol = x_sol

        self._reset_estado()

        if self.fitxer is not None and self.num is not None and self.prob is not None:
            self.read_dades(self.num, self.prob, self.fitxer)


    # Constructores alternativos ------------------------------------------------

    @classmethod
    def from_file(cls, fitxer: str, num: int, prob: int) -> "Simplex":
        """Crea una instancia leyendo los datos de un fichero de texto."""
        return cls(fitxer=fitxer, num=num, prob=prob)

    @classmethod
    def from_arrays(
        cls,
        cost: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        z_sol: Optional[float] = None,
        x_sol: Optional[list[int]] = None,
    ) -> "Simplex":
        """Crea una instancia a partir de los arrays del problema."""
        return cls(cost=cost, A=A, b=b, z_sol=z_sol, x_sol=x_sol)


    # Estado interno -----------------------------------------------------------

    def _reset_estado(self) -> None:
        """Reinicia todos los atributos de resultado antes de cada ejecución."""
        self.x: Optional[np.ndarray] = None
        self.z: Optional[float] = None
        self.estat: Optional[str] = None
        self.base_final: Optional[list[int]] = None
        self.B_inv_final: Optional[np.ndarray] = None
        self.r: Optional[np.ndarray] = None
        self.iteracions: int = 0
        self.iteraciones_log: list[dict] = []


    # Resolución principal ----------------------------------------------------------------

    def solve(self, tol: float = 1e-12) -> None:
        """
        Ejecuta Fase I + Fase II y guarda los resultados en el objeto.

        Retorna: (x, z_o_estado, base_final, B_inv_final, iteraciones_fase2)
        """
        if self.cost is None or self.A is None or self.b is None:
            raise ValueError("Faltan datos del problema: cost, A o b son None.")

        self._reset_estado()

        # --- Fase I ---
        base, estado_o_B_inv, iter_fase1 = self._fase_inicial(tol=tol)
        self.iteracions = iter_fase1

        # Si Fase I detecta infactibilidad o problema no acotado, salir sin error
        if isinstance(estado_o_B_inv, str):
            self.estat = estado_o_B_inv
            print(f"[Simplex] {estado_o_B_inv}: el problema no tiene solución factible.")
            return None, estado_o_B_inv, None, None, iter_fase1

        B_inv = estado_o_B_inv  # En caso contrario, es la B_inv

        # --- Fase II ---
        x, z_o_estado, base_final, B_inv_final, iter_fase2 = self._simplex_proces(
            basiques=base,
            inversa=B_inv,
            tol=tol,
        )

        # Guardar resultados
        self.x = x
        self.base_final = base_final
        self.B_inv_final = B_inv_final
        self.iteracions = iter_fase1 + iter_fase2

        if isinstance(z_o_estado, str):
            self.estat = z_o_estado
            self.z = None
        else:
            self.estat = "Óptimo"
            self.z = float(z_o_estado)

        # Costes reducidos finales
        if self.base_final is not None and self.B_inv_final is not None:
            cost_b = self.cost[self.base_final]
            self.r = self.cost - cost_b @ self.B_inv_final @ self.A

        self._print_resultados(fase="II")


    # Algoritmo del simplex (núcleo) --------------------------------------------------------------

    def _simplex_proces(
        self,
        basiques: list[int],
        inversa: np.ndarray,
        tol: float = 1e-12,
    ) -> tuple:
        """
        Ejecuta el simplex primal desde una base factible dada.
        Utiliza la regla de Bland para desempates.

        Retorna: (x_full, z_o_estado, base, B_inv, iteraciones_totales)
        """
        if self.cost is None or self.A is None or self.b is None:
            raise ValueError("Faltan datos del problema: cost, A o b son None.")

        cost = self.cost
        A = self.A
        b = self.b
        m, n = A.shape

        basiques = list(basiques)
        no_basiques = sorted(j for j in range(n) if j not in basiques)
        B_inv = inversa.astype(float).copy()

        x_B = B_inv @ b.astype(float)
        if np.any(x_B < -tol):
            raise ValueError("La base inicial no es factible (x_B < 0).")

        iteracio = self.iteracions

        while True:
            cost_b = cost[basiques]

            # Costes reducidos de las no básicas
            r_N = cost[no_basiques] - cost_b @ B_inv @ A[:, no_basiques]

            # Selección variable entrante (Bland: índice mínimo con r < 0)
            candidatos = [(no_basiques[e], e) for e in range(len(r_N)) if r_N[e] < -tol]
            if not candidatos:
                break  # Óptimo alcanzado

            entra_col, e = min(candidatos, key=lambda t: t[0])

            # Dirección de descenso
            d_B = -(B_inv @ A[:, entra_col])

            # Comprobar no acotado
            if np.all(d_B >= -tol):
                x_full = np.zeros(n)
                x_full[basiques] = x_B
                return x_full, "No acotado", basiques, B_inv, iteracio

            # Test de la razón (Bland en empates)
            ratios = [
                (-x_B[i] / d_B[i], basiques[i], i)
                for i in range(m)
                if d_B[i] < -tol
            ]
            theta, _, p = min(ratios, key=lambda t: (t[0], t[1]))

            var_surt = basiques[p]

            # Actualizar x_B
            x_B += theta * d_B
            x_B[np.abs(x_B) < tol] = 0.0
            x_B[p] = theta

            # Actualizar base
            basiques[p] = entra_col
            no_basiques = sorted(j for j in range(n) if j not in basiques)

            # Valor objetivo actual (para el log)
            x_full = np.zeros(n)
            x_full[basiques] = x_B
            z_actual = float(cost @ x_full)

            # Actualizar B_inv por pivotación eta
            T = np.eye(m)
            T[:, p] = [-d_B[i] / d_B[p] if i != p else -1.0 / d_B[p] for i in range(m)]
            B_inv = T @ B_inv

            iteracio += 1
            self.iteraciones_log.append({
                "iteracio": iteracio,
                "iout":     p,
                "q":        entra_col,
                "B_p":      var_surt,
                "theta":    theta,
                "z":        z_actual,
            })

        # Solución óptima
        x_full = np.zeros(n)
        x_full[basiques] = x_B
        z_val = float(cost @ x_full)

        return x_full, z_val, basiques, B_inv, iteracio


    # Fase I -------------------------------------------------------

    def _fase_inicial(self, tol: float = 1e-12) -> tuple:
        """
        Construye y resuelve el problema de Fase I para encontrar una SBF inicial.

        Retorna:
          - Si factible:     (base, B_inv, iteraciones_fase1)
          - Si infactible:   (None, "Infactible", iteraciones_fase1)
          - Si no acotado:   (None, "No acotado F1", iteraciones_fase1)
        """
        if self.A is None or self.b is None:
            raise ValueError("Faltan datos del problema: A o b son None.")

        A, b = self.A, self.b
        m, n = A.shape

        # Problema auxiliar: min sum(artificiales), s.a. [A | I] x = b
        A_f1 = np.hstack((A, np.eye(m)))
        cost_f1 = np.array([0.0] * n + [1.0] * m)

        aux = Simplex.from_arrays(cost=cost_f1, A=A_f1, b=b)

        x_f1, z1, base_f1, B_inv_f1, iter_f1 = aux._simplex_proces(
            basiques=list(range(n, n + m)),
            inversa=np.eye(m),
            tol=tol,
        )

        # Imprimir log de Fase I antes de retornar
        self.iteraciones_log = aux.iteraciones_log
        self._print_resultados(fase="I")
        self.iteraciones_log = []

        if float(z1) > tol:
            return None, "Infactible", iter_f1 
            # En la fase I el problema no puede dar no acotado, por lo tanto no cal contemplar el caso.

        # Eliminar artificiales de la base
        base = list(base_f1)
        B_inv_f1 = self._eliminar_artificiales(base, B_inv_f1, A_f1, n, tol)

        # Recomputar B_inv para el problema original (sin artificiales)
        try:
            B_inv = np.linalg.inv(A[:, base])
        except np.linalg.LinAlgError as e:
            raise ValueError("Base de Fase II no invertible.") from e

        x_B = B_inv @ b.astype(float)
        if np.any(x_B < -tol):
            raise ValueError("Base obtenida tras la Fase I no es factible.")

        return base, B_inv, iter_f1

    def _eliminar_artificiales(
        self,
        base: list[int],
        B_inv: np.ndarray,
        A_f1: np.ndarray,
        n: int,
        tol: float,
    ) -> np.ndarray:
        """
        Intenta sustituir las variables artificiales que han quedado en la base
        por variables originales mediante pivotación.
        """
        m = len(base)
        for fila in range(m):
            if base[fila] < n:
                continue  # Ya es una variable original

            # Buscar columna original para pivotar
            for j in range(n):
                if j in base:
                    continue
                u = B_inv @ A_f1[:, j]
                if abs(u[fila]) > tol:
                    base_ant = base[fila]
                    base[fila] = j
                    try:
                        B_inv = np.linalg.inv(self.A[:, base])
                        break
                    except np.linalg.LinAlgError:
                        base[fila] = base_ant
            else:
                raise ValueError("No se ha podido eliminar una variable artificial de la base.")

        return B_inv


    # Lectura de datos ----------------------------------------------------

    def read_dades(self, num: int, prob: int, fitxer: str) -> None:
        """Lee los datos del problema desde un fichero de texto."""
        with open(fitxer, "r", encoding="utf-8") as f:
            lines = f.readlines()

        idx = self._find_section_start(lines, num, prob) + 1
        cost, idx = self._parse_cost(lines, idx)
        A, idx = self._parse_A(lines, idx)
        b, idx = self._parse_b(lines, idx)
        z_sol, x_sol = self._parse_solution(lines, idx)

        self.cost = np.array(cost, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        self.z_sol = z_sol
        self.x_sol = x_sol


    # Métodos auxiliares de parsing (privados) ---------------------------------------------------------------

    @staticmethod
    def _extract_ints(line: str) -> list[int]:
        return list(map(int, re.findall(r"-?\d+", line)))

    @staticmethod
    def _is_number_line(line: str) -> bool:
        s = line.strip()
        return bool(re.match(r"^[\d\- ]+$", s)) and s != ""

    @staticmethod
    def _find_section_start(lines: list[str], num: int, prob: int) -> int:
        tag_num = f"datos{num}".lower()
        tag_prob = f"problemapl{prob}".lower()
        for i, line in enumerate(lines):
            normalized = line.replace(" ", "").lower()
            if tag_num in normalized and tag_prob in normalized:
                return i
        raise ValueError("No se ha encontrado la sección solicitada en el fichero.")

    def _parse_cost(self, lines: list[str], idx: int) -> tuple[list[int], int]:
        cost: list[int] = []
        while idx < len(lines) and "A=" not in lines[idx]:
            if self._is_number_line(lines[idx]):
                cost += self._extract_ints(lines[idx])
            idx += 1
        if idx >= len(lines):
            raise ValueError("No se ha encontrado el bloque A= después del coste.")
        return cost, idx

    def _parse_A(self, lines: list[str], idx: int) -> tuple[list[list[int]], int]:
        idx += 1  # Saltar la línea "A="
        blocks: list[list[list[int]]] = []
        current_block: list[list[int]] = []

        while idx < len(lines) and "b=" not in lines[idx]:
            line = lines[idx]
            if "Column" in line:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            elif self._is_number_line(line):
                current_block.append(self._extract_ints(line))
            idx += 1

        if current_block:
            blocks.append(current_block)
        if not blocks:
            raise ValueError("No se ha podido leer ninguna fila del bloque A=.")

        # Concatenar bloques por columnas
        A = [row[:] for row in blocks[0]]
        for block in blocks[1:]:
            if len(block) != len(A):
                raise ValueError("Bloques de A con número de filas inconsistente.")
            for i, row_part in enumerate(block):
                A[i] += row_part

        if idx >= len(lines):
            raise ValueError("No se ha encontrado el bloque b= después de A.")
        return A, idx

    def _parse_b(self, lines: list[str], idx: int) -> tuple[list[int], int]:
        idx += 1  # Saltar la línea "b="
        if idx >= len(lines):
            raise ValueError("Falta la línea con el vector b.")
        b = self._extract_ints(lines[idx])
        return b, idx + 1

    def _parse_solution(
        self, lines: list[str], idx: int
    ) -> tuple[Optional[float], Optional[list[int]]]:
        z_sol: Optional[float] = None
        x_sol: Optional[list[int]] = None

        while idx < len(lines):
            line = lines[idx]
            if "---" in line or "problema PL" in line.lower():
                break
            if "z*=" in line and "---" not in line:
                idx += 1
                if idx < len(lines):
                    z_sol = float(lines[idx].strip())
                idx += 3
                if idx < len(lines) and "---" not in lines[idx]:
                    x_sol = [x - 1 for x in self._extract_ints(lines[idx])]
                break
            idx += 1

        return z_sol, x_sol


    # Impresión de resultados -------------------------------------------------------

    def _print_resultados(self, fase: str = "II") -> None:
        """Imprime el log de iteraciones y, si es Fase II, la solución óptima."""
        print(f"[Simplex] Fase {fase}")
        for log in self.iteraciones_log:
            print(
                f"[Simplex] Iteración {log['iteracio']:3d} : "
                f"iout = {log['iout']}, "
                f"q = {log['q']}, "
                f"B(p) = {log['B_p']}, "
                f"theta* = {log['theta']:.3f}, "
                f"z = {log['z']:.3f}"
            )

        if fase == "II":
            if self.estat == "No acotado":
                print("\n[Simplex] Problema NO ACOTADO: la función objetivo no tiene mínimo finito.")
            elif self.estat == "Infactible":
                print("\n[Simplex] Problema INFACTIBLE: no existe solución factible.")
            elif self.estat == "Óptimo" and self.base_final is not None and self.r is not None:
                print("\nSolución óptima:")
                print(f"  vb = {' '.join(map(str, self.base_final))}")
                print(f"  xb = {' '.join(f'{v:.4f}' for v in self.x[self.base_final])}")
                print(f"  z  = {self.z:.4f}")
                print(f"  r  = {' '.join(f'{v:.4f}' for v in self.r)}")

                if self.z_sol is not None or self.x_sol is not None:
                    print("\nSolución esperada (del fichero):")
                    print(f"  z* = {self.z_sol}")
                    print(f"  vb*= {self.x_sol}")