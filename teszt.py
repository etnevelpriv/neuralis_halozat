# A numpy csomagot importoljuk, hogy tudjuk hasznalni a metodusait. Ez a csomag a komplex matematikai muveletekben lesz nagy segitseg
import numpy as np

# A szukseges adathalmazokat letrehozzuk
# Bemeneti adathalmaz. XOR kapu lehetosegei egy tombben elmentve. Ezt az adathalmazt jelenleg matrixkent tekintjuk, amely vektorokra van bontva. Minden sor egy vektor es a vektorok elemei a bemeneti neuronok ertekei
x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
# Target ertekeket mentjuk el egy adathalmazban. Minden bemeneti vektorhoz rendelunk egy target erteket. A neuralis halonak a celja, hogy megtanulja ezeket es hasonlo ertekeket tudjon produkalni
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Deklarajuk a konstans skalar ertekeinket es erteket rendelunk hozzajuk
learning_rate = 0.1 # Tanulasi rata megadja, hogy mennyire gyorsan tanuljon az adott folyamat alatt a neuralis halo. Ha az ertek tul magas, akkor pontatlan is lehet, ha tul alacsony, akkor tul lassan fog "tanulni"(=tul lassan valtoztat erteket)
epochs = 10000 # Epochs megadja hogy az egyes folyamat hanyszor fusson le (=iteraciok szama)

# Letrehozzuk a sulyvektorokat es a bias-okat
# A weight azt mutatja meg, hogy az egyes bemeneti adatok (neuronok) mennyire szamitanak
# Ha sulyvektort es bemeneti vektort osszeszoroznank (skalaris szorzat), akkor egy skalar erteket kapnank.
# Viszont jelenleg egy bemeneti matrixrol (amely vektorokra van tagolva) es egy sulymatrixrol (amely vektorokra van tagolva) van szo
# Igy egy matrixszorzast hajtunk vegre es a kapott eredmenyunk egy matrix (nem fontos tudni, hogy hogyan mukodik a matrixok szorzasa jelenleg)
# A matrixszorzat eredmenyeben minden szam egy adott rejtett neuron kimeneti erteke egy adott bemenetre
# A bias azt az erteket (akar skalar, akar vektor) jelenti, amennyivel a matrixszorzat(vagy skalarszorzat, de jelen esetben matrixszorzat) minden vektoranak minden elemet (rejtett neuron kimeneti erteket) eltoljuk.
# bias = neurononkénti eltolás

# A bemeneti reteg es a rejtett reteg kozotti atmenetet biztosito valotzok
weights_input_hidden = np.random.rand(2,2) # Beepitett metodus, n=2 es k=2 matrixot hoz letre random szamokkal
bias_input_hidden = np.zeros((1,2)) # Beepitett metodus, n=1 es k=2 nullmatrixot hoz letre

# A rejtett reteg es a kimeneti reteg kozotti atmenetet biztosito valtozok
weights_output_hidden = np.random.rand(2,1)
bias_output_hidden = np.zeros((1,1))

# Sigmoid aktivacios fuggveny letrehozasa
# A neuralis halo egyik fontos resze az aktivacios fuggveny
# Tobb fele lehet, de most a sigmoidot fogjuk hasznalni
# Barmilyen szamot kepes atalakitani 0 es 1 koze
# A mogottes matekot most nem kell erteni, eleg ennyit tudni:
#   ha a szam nagyon negativ akkor az aktivacios fuggveny utan 0-hoz kozel lesz
#   ha a szam nagyon pozitov akkor az aktivacios fuggveny utan 1-hez kozel lesz
#   ha a szam kozepes akkor az aktivacios fuggveny utan 0 es 1 kozott lesz (de egyikhez sem tul kozel) lesz
def calcSigmoid(x):
    return ( 1 / (1 + np.exp(-x)) )

'''
Konkret pelda vegigvezetve:

# Bemeneti matrix (2 oszlop es 4 sor)
x = [
 [0,0],
 [0,1],
 [1,0],
 [1,1]
]
A bemeneti matrixbol a neuronok minden egyes vektornak az elemei. A vektor elemenek az indexe megegyezeik a neuron indexevel.
[0,0] -> Neuron1=0, Neuron2=0
[0,1] -> Neuron1=0, Neuron2=1
[1,0] -> Neuron1=1, Neuron2=0
[1,1] -> Neuron1=1, Neuron2=1

# Target ertek matrix a bemeneti matrix es a XOR kapu alapjan
y = [
 [0],
 [1],
 [1],
 [0]
]

# Sulymatrix (input)
weights_input_hidden (2 oszlop es 2 sor) = [
 [0.5, -1.0],
 [1.5,  2.0]
]

# Bias (input)
bias_input_hidden (2 oszlop 1 sor) = [0.1, -0.2]

Szorzas menete (x es weights_input_hidden szorzata): (
1.Bemenet:
    Neuron 1:
    (0 * 0.5) + (0 * 1.5) = 0
    Neuron 2:
    (0 * -1.0) + (0 * 2.0) = 0
    Eredmeny: [0, 0]
2.Bemenet:
    Neuron 1:
    (0 * 0.5) + (1 * 1.5) = 1.5
    Neuron 2:
    (0 * -1.0) + (1 * 2.0) = 2.0
    Eredmeny: [1.5, 2.0]
3.Bemenet:
    Neuron 1:
    (1 * 0.5) + (0 * 1.5) = 0.5
    Neuron 2:
    (1 * -1.0) + (0 * 2.0) = -1.0
    Eredmeny: [0.5, -1.0]
4.Bemenet:
    Neuron 1:
    (1 * 0.5) + (1 * 1.5) = 2.0
    Neuron 2:
    (1 * -1.0) + (1 * 2.0) = 1.0
    Eredmeny: [2.0, 1.0]`

# A matrixszorzas teljes eredmenye ez (2 oszlop es 4 sor):
hidden_input = [
 [0.0,  0.0],
 [1.5,  2.0],
 [0.5, -1.0],
 [2.0,  1.0]
]

# A matrixszorzas eredmenyeben az oszlopok a rejtett neuronok kimeneteit tartalmazzak (2db):
[
    0.0,
    1.5,
    0.5,
    2.0
 ],
 [
    0.0,
    2.0,
    -1.0,
    1.0
 ]
 
# A matrix szorzas utan hozzaadjuk a bias-t.
Az osszeadas menete:
[0.0, 0.0] + [0.1, -0.2] = [0.1, -0.2]
[1.5, 2.0] + [0.1, -0.2] = [1.6, 1.8]
[0.5, -1.0] + [0.1, -0.2] = [0.6, -1.2]
[2.0, 1.0] + [0.1, -0.2] = [2.1, 0.8]

# A bias input eredmenye:
hidden_input_with_bias = [
 [0.1, -0.2],
 [1.6,  1.8],
 [0.6, -1.2],
 [2.1,  0.8]
]

# A bias input eredmenyet betesszuk egy aktivacios fuggvenybe (sigmoid).
sigmoid(x) ≈
0.1  -> 0.525
-0.2 -> 0.450
1.6  -> 0.832
1.8  -> 0.858
0.6  -> 0.646
-1.2 -> 0.231
2.1  -> 0.891
0.8  -> 0.690

# A sigmoid fuggveny visszateritesi erteke egy matrix lesz
hidden_output = [
 [0.525, 0.450],
 [0.832, 0.858],
 [0.646, 0.231],
 [0.891, 0.690]
]

# Sulymatrix (output)
weights_output_hidden (1 oszlop es 2 sor) = [
 [1.0],
 [-1.5]
]

# Bias (output)
bias_output_hidden (1 oszlop es 1 sor {skalar ertek}) = [0.2]

# A sigmoid fuggveny visszateritesi erteket felhasznaljuk az output retegunk soran
# A logika ugyan az, de a bemeneti adathalmaz nem az 'x', hanem a sigmoid fuggveny visszateritesi erteke, a 2 oszlopos input weight matrix helyett pedig az 1 oszlopos output weight matrixot hasznaljuk annak erdekeben, hogy a kimenetunk 1 oszlopbol alljon
# A szorzas menete (hidden_output es weights_hidden_output szorzata) es a bias hozzaadasa:
1.bemenet:
    (0.525 * 1.0) + (0.450 * -1.5)
    = 0.525 - 0.675
    = -0.15
    + bias:
    -0.15 + 0.2 = 0.05
2.bemenet:
    (0.832 * 1.0) + (0.858 * -1.5)
    = 0.832 - 1.287
    = -0.455
    + bias:
    -0.455 + 0.2 = -0.255
3.bemenet:
    (0.646 * 1.0) + (0.231 * -1.5)
    = 0.646 - 0.346
    = 0.300
    + bias:
    0.300 + 0.2 = 0.5
4. bemenet:
    (0.891 * 1.0) + (0.690 * -1.5)
    = 0.891 - 1.035
    = -0.144
    + bias:
    -0.144 + 0.2 = 0.056
# Tehat a matrixszorzas, majd a bias hozzaadas eredmenye:
final_input = [
 [ 0.05 ],
 [-0.255],
 [ 0.5 ],
 [ 0.056]
]

# A kapott eredmmenyt is bekell tenni az aktivacios sigmoid fuggvenybe (semmi nem valtozik, csak a bemeneti adatkent kapott parameter logikusan)
0.05   -> 0.512
-0.255 -> 0.436
0.5    -> 0.622
0.056  -> 0.514

# A vegso kapott visszaterites ertek lesz az eredmenyunk, kesobb ezen javitunk annak erdekeben, hogy elerjuk a kivant target ertekeket
final_output = [
 [0.512],
 [0.436],
 [0.622],
 [0.514]
]
'''

# Neuralis halo tanitasa
# Az epochs szam alapjan ennyiszer fog vegigfutni a tanulasi folyamat.
# Minden egyes iteracioban ezek a dolgok tortennek idorendi sorrendben:
#   1. Kiszamolja a rejtett reteg bemenetet (bemeneti adathalmaz es rejtett reteg kozotti atmenet az 'x' bemeneti adathalmaz konstans matrixaval, a 'weights_input_hidden' sulyvektorok folyamatosan valtozo (epochs iteracioban modositjuk minden korben) matrixaval es a 'bias_input_hidden' folyamatosan valtozo (epochs iteracioban modositjuk minden korben) bias matrixaval)
#   2. Elvegzi a rejtett reteg bemeneti adahalmazanak aktivaciojat (ahol a sigmoid fuggveny parametere a rejtett reteg bemeneti adathalmaza)
#   3. Kiszamolja a rejtett reteg kimenetet (az elozo lepes visszateritesi erteke a bemeneti adathalmaz, tovabba felhasznaljuk a 'weights_output_hidden' sulyvektorok folyamatosan valtozo matrixat es a 'bias_output_hidden' folyamatosan valtozo bias matrixaval)
#   4. Elvegzi a rejtett reteg kimeneti aktivaciojat (ahol a sigmoid fuggveny parametere a rejtett reteg kimeneti adathalmaza)
#   5. Osszehasonlitja a neuralis halo vegso kimeneti adathalmazat (final_output) a target ertekek matrixaval (y) es megnezi, hogy mekkorat tevedett es egy valtozoban (err) elmentjuk az ertekeket
#   6. A vegso kimeneti adathalmazt (final_output) es a rejtett reteget (weights_output_hidden) atalakitjuk (derivaljuk) es elmentjuk oket egy-egy uj valtozoba (d_output es d_hidden) annak erdekeben, hogy megnezzuk a hiba iranyat es merteket, ezaltal tudunk csak atmenni a kovetkezo lepesre
#   7. A hiba alapjan visszafele modositjuk a valtozokat (weight matrixok es bias matrixok).
#   8. Minden szazadik iteraciot kiiratunk, hogy lassuk a valtoztatasokat es ezzel segitjuk a hibakezelest, tovabba segit ertelmezni a neuralis halonk tanulasi folyamatat
# Az elso 4 lepest hivjuk 'forward pass'-nek, mivel itt fut le a rejtett reteg, itt tekintjuk meg a kimenetet annak erdekeben, hogy tanithassuk es elemezhessuk neuralis halonkat
# Az 5. lepes a hiba szamitasa
# A 6. lepes a 'backpropagation/visszaterjesztes', hisz itt elemezzuk a hibat visszamenoleg
# A 7. lepes a valtozok modositasa
# A 8. lepes a logolas (nem kotelezo, de hasznos)
for epoch in range(epochs):
    # 1.
    hidden_input = np.dot(x, weights_input_hidden) + bias_input_hidden
    # 2.
    hidden_output = calcSigmoid(hidden_input)
    # 3.
    final_input = np.dot(hidden_output, weights_output_hidden) + bias_output_hidden
    # 4.
    final_output = calcSigmoid(final_input)
    # 5.
    err = y - final_output
    # 6. (nem kell erteni, a lenyeg, hogy matek es ezekkel a muveletekkel tudjuk kiszamolni, hogy a kimenetek mennyire erzekenyek a valtozasokra)
    d_output = err * (final_output * (1 - final_output))
    d_hidden = d_output.dot(weights_output_hidden.T) * (hidden_output * (1 - hidden_output))
    # 7. (itt tortenik maga a tanulas, szinten nem kell erteni, emelt matek)
    weights_output_hidden += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += x.T.dot(d_hidden) * learning_rate
    bias_output_hidden += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    bias_input_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
    # 8.
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}\nError: {err}\nFinal output: {final_output}")

# Vegul utoljara lefuttatjuk a halot (mar valtoztatas es tanitas nelkul), hogy megnezzuk, hogy mit tanult
hidden_input = np.dot(x, weights_input_hidden) + bias_input_hidden
hidden_output = calcSigmoid(hidden_input)
final_input = np.dot(hidden_output, weights_output_hidden) + bias_output_hidden
final_output = calcSigmoid(final_input)
print(final_output)