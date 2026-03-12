# Nagyon regen programoztam pythopnban, nem tudnam magamtol elkesziteni a projektet, tutorial alapjan megyek vegig.
# Forras: https://realpython.com/python-ai-neural-network/
input_vector = [1.72, 1.23]

weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product1 = first_indexes_mult + second_indexes_mult

print(dot_product1)