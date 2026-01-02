
# Ejemplo simple de enumerate
frutas = ['manzana', 'banana', 'cereza']

# Con enumerate (más limpio/Pythonico)
print("--- Usando enumerate ---")
for i, fruta in enumerate(frutas):
    print(f"Índice: {i}, Fruta: {fruta}")

# Salida esperada:
# Índice: 0, Fruta: manzana
# Índice: 1, Fruta: banana
# Índice: 2, Fruta: cereza

# También puedes empezar a contar desde otro número
print("\n--- Empezando desde 1 ---")
for i, fruta in enumerate(frutas, start=1):
    print(f"Número: {i}, Fruta: {fruta}")
