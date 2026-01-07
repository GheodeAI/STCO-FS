import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def benchmark_cross_val_jobs():
    # 1. Generar datos sintéticos
    # Usamos un dataset lo suficientemente grande para que la paralelización valga la pena
    print("Generando datos sintéticos (10000 muestras, 50 características)...")
    X, y = make_classification(n_samples=10000, n_features=50, n_informative=20, random_state=42)
    
    # 2. Definir el clasificador
    # Usamos n_jobs=1 en el clasificador para aislar el efecto de paralelizar el bucle de validación cruzada
    clf = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)
    
    print(f"Clasificador: {clf.__class__.__name__}")
    
    # 3. Benchmark n_jobs = 1 (Secuencial)
    print("\nEjecutando cross_val_score con n_jobs=1 (Secuencial)...")
    start_time = time.time()
    scores_1 = cross_val_score(clf, X, y, cv=5, n_jobs=1)
    end_time = time.time()
    duration_1 = end_time - start_time
    print(f"Tiempo con n_jobs=1: {duration_1:.4f} segundos")
    
    # 4. Benchmark n_jobs = -1 (Paralelo, usa todos los cores)
    print("\nEjecutando cross_val_score con n_jobs=-1 (Paralelo)...")
    start_time = time.time()
    scores_2 = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
    end_time = time.time()
    duration_2 = end_time - start_time
    print(f"Tiempo con n_jobs=-1: {duration_2:.4f} segundos")
    
    # 5. Comparación de resultados
    if duration_2 > 0:
        speedup = duration_1 / duration_2
        print(f"\nMejora de velocidad (Speedup): {speedup:.2f}x")
    
    if duration_2 < duration_1:
         print("Conclusión: n_jobs=-1 fue más rápido.")
    else:
         print("Conclusión: n_jobs=-1 no mejoró el tiempo (puede deberse a overhead de procesos o dataset muy pequeño).")

if __name__ == "__main__":
    benchmark_cross_val_jobs()
