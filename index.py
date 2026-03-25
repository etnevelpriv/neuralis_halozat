# Nagyon regen programoztam pythopnban, nem tudnam magamtol elkesziteni a projektet, tutorial alapjan megyek vegig.
# Forras: https://realpython.com/python-ai-neural-network/

# Projekthez szukseges hattertudas :
# Linearis algebra: Vektorok, azok mukodese, a skalaris szorzat, matrixok
# Calculus: Hatarertek, differencialszamitas, derivalas, gradiens, parcialis derivalas, lancszabaly, euler szam, sigmoid fuggveny

# Projekthez kapcsolodo altalanos fogalmak
# Skalaris szorzat: Ket vektorhoz egy valos szamot rendel. Megmutatja, hogy az egyik vektor hogyan viszonyul a masikhoz. A 2 vektornak ugyan olyan hosszunak kell lennie (jelen esetben ez azt jelenti, hogy ugyan annyi adatbol kell allnia a tombnek). Ha 0 az eredmeny, akkor merolegesek egymasra a vektorok.

"""
Elkeszitjuk a bemeneti adatok listajat es a sulyok listajat
Az input vektor es a sulyvektorbol keszitunk skalaris szorzatot, majd hozzaadjuk a bias-t, amit elmentunk a dot productba. (elso layer)
Elvegezzuk a sigmoid egyenletet (masodik layer)
A masodik layer visszateritesi erteke a prediction maga, ezt elmentjuk egy valtozoba
"""

# Matematikai muveletek szamitasat vagyunk kepesek vegezni a numpy csomag beepitett fuggvenyeivel.
import numpy as np

# Bemeneti adatok tombje, amit kepesek vagyunk vektorkent ertelmezni jelen esetben.
# input_vector = [1.72, 1.23]
input_vector = np.array([1.72, 1.23]) # numpy array megkonnyiti a muveleteket. Gyorsabb, tovabba matrix muveletekre konnyebben lehet alkalmazni

# Sulyvektorok. Azt mutatjak meg, hogy mennyire mervado a bemeneti adat erteke. Ha a bemeneti tomb valamelyik adatahoz 0 erteku sulyt csatolunk, akkor az a bemeneti ertek nem fog szamitani.
# weights_1 = [1.26, 0]
# weights_2 = [2.17, 0.32]
weights_1 = np.array([1.26, 0])
weights_2 = np.array([2.17, 0.32])

# Bias az az ertek, amely segit, hogy a modell ne csak az origo kozelebe tudjon loni, hisz ezzel toljuk el
bias = np.array([0.0])

# Elvart kimenetet athuztam ide, hisz konstans
target = 0

# A kommentezett resz ugyan azt csinalja, mint az importalt csomag beepitett fuggvenye. Itt manualis szamoljuk a skalaris szorzatot, az np.dot pedig beepitetten teszi ugyan ezt a megadott array-ekbol.
"""
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product1 = first_indexes_mult + second_indexes_mult
"""

# Skalaris szorzat ertekenek szamitasa. A bemeneti adatok es a sulyvektor osszefuggeset mutatja minel inkabb tavolodik a 0-tol, annal nagyobb az osszefugges (annal inkabb parhuzamos a 2 vektor).
# dot_product1 = np.dot(input_vector, weights_1)
# dot_product2 = np.dot(input_vector, weights_2)

# Ugyan ugy skalaris szorzat szamitasa, csak fuggvenybe rakva, biast is hozzaadja
def calcDotProduct(input_vector, weights_1, bias):
    return (np.dot(input_vector, weights_1) + bias)

# Pontos matematikai hatteret nem tudom, mert nem tanultam calculust sohasem, viszont tudom, hogy valoszinuseg szamitashoz lehet alkalmazni, hisz 0 es 1 koze teszi az erteket
def calcSigmoid(x):
    return ( 1 / (1 + np.exp(-x)) )

# Ez az egy fuggveny futtatja le az egeszet, csak meghivja a tobbi fuggvenyt es layerkent kezeli azokat
def makePrediction(input, weight, bias):
    layer_1 = calcDotProduct(input, weight, bias)
    layer_2 = calcSigmoid(layer_1)
    return layer_2

# A vart kimenetet adjuk meg es megmerjuk a hiba merteket.
def calcErr(prediction, target):
    base_err = prediction - target
    mse = np.square(base_err) # A hiba merteket negyzetre emeljuk, igy mindig pozitiv lesz a hiba, es a nagy hibakat sokkal jobban bunteti, mig a kicsiket lekicsinyiti
    print(f"Prediction: {prediction}; Error: {mse}")
    # Ha "prediction - target" az kisebb, mint 0, akkor novelni kell az erteket, ha nagyobb, akkor csokkenteni kell. 0-hoz kozeli allapot a megfelelo.
    # Annak erdekeben, hogy tudjuk melyik iranyba kell ezt az erteket novelni, derivalni kell. En nem tudok derivalni, de a megadott oldal tokeletesen leirjam hogy mit kell tudni. : ' the derivative of xⁿ is nx⁽ⁿ⁻¹⁾ '. Ebbol kovetkezik , hogy jelen esetunkben a hiba derivalt erteke 2 * (prediction - target)    
    derivated_value = 2 * base_err
    print(f"The derivated value of the error: {2 * (base_err)}")
    if derivated_value > 0.02:
        weights_1[0] = weights_1[0] - 0.02
        prediction = makePrediction(input_vector, weights_1, bias)
        calcErr(prediction, target)
    elif derivated_value < -0.02:
        weights_1[0] = weights_1[0] - 0.02
        prediction = makePrediction(input_vector, weights_1, bias)
        calcErr(prediction, target)
    else:
        print('The prediction is successfull')

# Elmentjuk egy valtozoba a predictiont
prediction = makePrediction(input_vector, weights_1, bias)
calcErr(prediction, target)