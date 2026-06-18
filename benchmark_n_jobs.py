import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def benchmark_cross_val_jobs():
    print("Generando datos sintéticos (20000 muestras, 40 características)...")
    # Aumentamos un poco el tamaño para asegurar que se note la diferencia
    X, y = make_classification(n_samples=20000, n_features=40, n_informative=20, random_state=42)
    
    # Usamos n_jobs=1 en el modelo para que la paralelización sea solo por cross_val_score
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)
    
    print(f"\nModelo: {clf.__class__.__name__} (n_estimators=100, n_jobs=1)")
    print("-" * 50)

    # Benchmark n_jobs = 1
    print("Ejecutando cross_val_score con n_jobs=1 (Secuencial)...")
    start_time = time.time()
    scores_1 = cross_val_score(clf, X, y, cv=5, n_jobs=1)
    end_time = time.time()
    duration_1 = end_time - start_time
    print(f"-> Tiempo: {duration_1:.4f} segundos")
    print(f"-> Scores: {scores_1}")

    print("-" * 50)

    # Benchmark n_jobs = -1
    print("Ejecutando cross_val_score con n_jobs=-1 (Usa todos los núcleos)...")
    start_time_2 = time.time()
    scores_2 = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    end_time_2 = time.time()
    duration_2 = end_time_2 - start_time_2
    print(f"-> Tiempo: {duration_2:.4f} segundos")
    print(f"-> Scores: {scores_2}")

    print("-" * 50)
    
    # Resultados
    if duration_2 > 0:
        speedup = duration_1 / duration_2
        print(f"Mejora de velocidad (Speedup): {speedup:.2f}x")
        
    if duration_2 < duration_1:
         print("Conclusión: Utilizar n_jobs=-1 redujo el tiempo de ejecución.")
    else:
         print("Conclusión: No hubo mejora de tiempo (posible overhead o dataset pequeño).")

if __name__ == "__main__":
    benchmark_cross_val_jobs()
