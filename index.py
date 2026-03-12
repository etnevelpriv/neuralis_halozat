# Nagyon regen programoztam pythopnban, nem tudnam magamtol elkesziteni a projektet, tutorial alapjan megyek vegig.
# Forras: https://realpython.com/python-ai-neural-network/

# Projekthez kapcsolodo altalanos fogalmak
# Skalaris szorzat: Ket vektorhoz egy valos szamot rendel. Megmutatja, hogy az egyik vektor hogyan viszonyul a masikhoz. A 2 vektornak ugyan olyan hosszunak kell lennie (jelen esetben ez azt jelenti, hogy ugyan annyi adatbol kell allnia a tombnek). Ha 0 az eredmeny, akkor merolegesek egymasra a vektorok.

# Matematikai muveletek szamitasat vagyunk kepesek vegezni a numpy csomag beepitett fuggvenyeivel.
import numpy as np

# Bemeneti adatok tombje, amit kepesek vagyunk vektorkent ertelmezni jelen esetben.
input_vector = [1.72, 1.23]

# Sulyvektorok. Azt mutatjak meg, hogy mennyire mervado a bemeneti adat erteke. Ha a bemeneti tomb valamelyik adatahoz 0 erteku sulyt csatolunk, akkor az a bemeneti ertek nem fog szamitani.
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

# A kommentezett resz ugyan azt csinalja, mint az importalt csomag beepitett fuggvenye. Itt manualis szamoljuk a skalaris szorzatot, az np.dot pedig beepitetten teszi ugyan ezt a megadott array-ekbol.
"""
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product1 = first_indexes_mult + second_indexes_mult
"""

# Skalaris szorzat ertekenek szamitasa. A bemeneti adatok es a sulyvektor osszefuggeset mutatja minel inkabb tavolodik a 0-tol, annal nagyobb az osszefugges (annal inkabb parhuzamos a 2 vektor).
dot_product1 = np.dot(input_vector, weights_1)
dot_product2 = np.dot(input_vector, weights_2)  

print(dot_product1, dot_product2)