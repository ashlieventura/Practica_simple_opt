from simplex_cls import Simplex
import numpy as np
import sys
from io import StringIO

if __name__ == "__main__":
    output_file = open("resultados_simplex_Ashlie.txt", "w", encoding="utf-8")
    original_stdout = sys.stdout
    
    for prob_num in [1, 2, 3, 4]:
        msg = f"{'='*50}\nProblema {prob_num}\n{'='*50}"
        print(msg)
        output_file.write(msg + "\n")
        output_file.flush()
        
        # Capturar output de solve()
        buffer = StringIO()
        sys.stdout = buffer
        
        s = Simplex("OPT25-26_Datos práctica 1.txt", num=46, prob=prob_num)
        s.solve()
        
        output = buffer.getvalue()
        sys.stdout = original_stdout
        
        print(output, end='')
        output_file.write(output)
        
        print()
        output_file.write("\n")
    
    output_file.close()
    print("Resultados guardados en 'resultados_simplex_Ashlie.txt'")