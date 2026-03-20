from simplex_cls import Simplex
import numpy as np

if __name__ == "__main__":
    s = Simplex("OPT25-26_Datos práctica 1.txt", num=38, prob=4)
    x, z_or_status, base_final, B_inv_final, it = s.solve()

