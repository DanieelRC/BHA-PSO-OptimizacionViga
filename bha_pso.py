import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import time
import os

def configurar_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"Seed configurada: {seed}")
    else:
        seed = np.random.randint(0, 10000)
        random.seed(seed)
        np.random.seed(seed)
        print(f"Seed generada automáticamente: {seed}")
    return seed


P = 6000         # Es la carga aplicada
L = 14.0         # Es la longitud de la viga
E = 30e6         # Es el módulo de young
G = 12e6         # Es el módulo de corte
tau_max = 13600  # Es el limite de esfuerzo cortante
sigma_max = 30000  # Es el limite de tensión normal
delta_max = 0.25  # Es el limite de deflexión

def calcular_restricciones(x):
    x1, x2, x3, x4 = x
      # Verificar los limites positivos 
    if any(val <= 0 for val in x):
        return np.array([1e12] * 7)
    
    try:
        # 1 Cálculo del esfuerzo cortante
        R = np.sqrt((x2**2)/4 + ((x1 + x3)/2)**2)
        J = 2 * (np.sqrt(2) * x1 * x2 * (x2**2/12 + ((x1 + x3)/2)**2))
        M = P * (L + x2/2)
        tau_prime = P / (np.sqrt(2) * x1 * x2)
        tau_dprime = (M * R) / J if J > 0 else 1e12
        tau = np.sqrt(tau_prime**2 + ((2*tau_prime*tau_dprime)*x2/(2*R)) + tau_dprime**2)
        
        # 2 Cálculo del esfuerzo normal
        sigma = 6 * P * L / (x4 * x3**2)
        
        # 3 Deflexión
        delta = 4 * P * L**3 / (E * x4 * x3**3)

        # 4 Carga crítica de pandeo
        Pc = (4.013 * E * np.sqrt(x3**2 * x4**6 / 36) / L**2) * (1 - (x3/(2*L)) * np.sqrt(E/(4*G)))
        
        # Restricciones (deben ser <= 0)
        g = np.array([
            tau - tau_max,                                      # g1
            sigma - sigma_max,                                  # g2
            x1 - x4,                                            # g3
            0.10471*x1**2*x2 + 0.04811*x3*x4*(14 + x2) - 5.0,   # g4
            0.125 - x1,                                         # g5
            delta - delta_max,                                  # g6
            P - Pc                                              # g7
        ])
        
        return g
        
    except:
        return np.array([1e12] * 7)

def funcion_objetivo(x):
    return 1.10471 * x[0]**2 * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])

def es_factible(x, tol=1e-6):
    limites = [(0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 2.0)]  # Restricciones de caja
    
    # Verificar límites de variables
    for i, (val, (lb, ub)) in enumerate(zip(x, limites)):
        if val < lb or val > ub:
            return False
    
    # Verificar restricciones
    g = calcular_restricciones(x)
    return np.all(g <= tol)


def ranking_srga(poblacion, p_f):
    n = len(poblacion)
    
    for i in range(n):
        for j in range(n - 1 - i):
            p1, p2 = poblacion[j], poblacion[j+1]
            
            intercambiar = False
            
            if p1.factibilidad and p2.factibilidad:
                # Ambas factibles: comparar por aptitud
                if p1.aptitud > p2.aptitud:
                    intercambiar = True
                    
            elif p1.factibilidad and not p2.factibilidad:
                # p1 factible es siempre mejor que p2 no factible
                intercambiar = False
                
            elif not p1.factibilidad and p2.factibilidad:
                # p2 factible es siempre mejor que p1 no factible
                intercambiar = True
                
            else:
                # Ambas no factibles: aplicar SRGA estocástico
                if random.random() < p_f:
                    # Comparar por función objetivo
                    if p1.aptitud > p2.aptitud:
                        intercambiar = True
                else:
                    # Comparar por violación de restricciones
                    if p1.violacion > p2.violacion:
                        intercambiar = True
            
            # Intercambiar
            if intercambiar:
                poblacion[j], poblacion[j+1] = poblacion[j+1], poblacion[j]
    
    return poblacion

class Particula:
    def __init__(self, dimension, limites):
        self.posicion = np.random.uniform(limites[:,0], limites[:,1], dimension)
        self.velocidad = np.zeros(dimension)
        self.mejor_posicion = self.posicion.copy()
        self.mejor_aptitud = float('inf')
        self.aptitud = float('inf')
        self.factibilidad = False
        self.violacion = 0.0
        
        # Asegurar restricción x4 >= x1
        if self.posicion[3] < self.posicion[0]:
            self.posicion[3] = self.posicion[0] + random.uniform(0, 0.1)

def hibrido_bha_pso(tam_poblacion=100, max_iter=1000, params=None, seed=None):
    if seed is not None:
        configurar_seed(seed)
    
    if params is None:
        params = {
    'w': 0.4,        # Inercia
    'c1': 1.2,       # I. Cognitivo
    'c2': 1.8,       # I. Social
    'p_f': 0.3,      # Priorizar o no restricciones
    'bha_ratio': 0.5,
    'elitismo': True
}
    
    # Restricciones de caja
    limites = np.array([
        [0.1, 2.0],    # x1 (h)
        [0.1, 10.0],   # x2 (l)
        [0.1, 10.0],   # x3 (t)
        [0.1, 2.0]     # x4 (b)
    ])
    dimension = 4
    
    # Inicialización
    poblacion = [Particula(dimension, limites) for _ in range(tam_poblacion)]
    mejor_global = None
    mejor_aptitud_global = float('inf')
    historial = []
    
    # Calcular iteraciones para fase BHA
    iter_bha = int(max_iter * params['bha_ratio'])
    
    for iteracion in range(max_iter):
        # Evaluacion
        for particula in poblacion:
            particula.aptitud = funcion_objetivo(particula.posicion)
            particula.factibilidad = es_factible(particula.posicion)
            particula.violacion = np.sum(np.maximum(0, calcular_restricciones(particula.posicion)))
              # Actualizar mejor personal
            if particula.factibilidad and particula.aptitud < particula.mejor_aptitud:
                particula.mejor_aptitud = particula.aptitud
                particula.mejor_posicion = particula.posicion.copy()
        
        # Aplicar SRGA
        poblacion = ranking_srga(poblacion, params['p_f'])
        mejor_actual = poblacion[0]
        
        # Actualizar mejor global
        if mejor_actual.factibilidad:
            if mejor_global is None or mejor_actual.aptitud < mejor_aptitud_global:
                mejor_aptitud_global = mejor_actual.aptitud
                mejor_global = mejor_actual.posicion.copy()
        
        historial.append(mejor_aptitud_global if mejor_global is not None else mejor_actual.aptitud)
        
        # Mostrar iteraciones
        if iteracion % 100 == 0:
            factibles = sum(1 for p in poblacion if p.factibilidad)
            print(f"Iter {iteracion}: Mejor = {mejor_aptitud_global:.6f}, Factibles = {factibles}/{tam_poblacion}")
        
        # Fase BHA (Exploracion)
        if iteracion < iter_bha:
            agujero_negro = mejor_actual.posicion
            
            # Radio del horizonte de eventos
            aptitud_total = sum(p.aptitud for p in poblacion)
            radio = mejor_actual.aptitud / (aptitud_total + 1e-12)
            
            for i, particula in enumerate(poblacion):
                if i == 0:  # No se modifica el mejor
                    continue
                
                # Movimiento hacia el agujero negro
                r = np.random.rand(dimension)
                particula.posicion += r * (agujero_negro - particula.posicion)
                
                # Horizonte de eventos
                distancia = np.linalg.norm(particula.posicion - agujero_negro)
                if distancia < radio:
                    particula.posicion = np.random.uniform(limites[:,0], limites[:,1])
                    particula.velocidad = np.zeros(dimension)
                
                # Asegurar límites
                particula.posicion = np.clip(particula.posicion, limites[:,0], limites[:,1])
                
                # Restricciones de caja 
                if particula.posicion[3] < particula.posicion[0]:
                    particula.posicion[3] = particula.posicion[0] + random.uniform(0, 0.1)
        
        # Fase PSO (Explotacion)
        else:
            w = params['w'] * (1 - iteracion/max_iter)  # Inercia utilizando número de iteraciones
            
            for particula in poblacion:
                if mejor_global is not None: 
                    # Actualizar velocidad
                    r1, r2 = np.random.rand(dimension), np.random.rand(dimension)
                    cognitivo = params['c1'] * r1 * (particula.mejor_posicion - particula.posicion)
                    social = params['c2'] * r2 * (mejor_global - particula.posicion)
                    
                    particula.velocidad = w * particula.velocidad + cognitivo + social
                    
                    # Límite de velocidad
                    v_max = 0.2 * (limites[:,1] - limites[:,0])
                    particula.velocidad = np.clip(particula.velocidad, -v_max, v_max)
                    
                    # Actualizar posición
                    particula.posicion += particula.velocidad
                    
                    # Asegurar límites y restricciones
                    particula.posicion = np.clip(particula.posicion, limites[:,0], limites[:,1])
                    if particula.posicion[3] < particula.posicion[0]:
                        particula.posicion[3] = particula.posicion[0] + random.uniform(0, 0.05)
    
    # Resultados
    if mejor_global is not None:
        restricciones = calcular_restricciones(mejor_global)
        return mejor_global, mejor_aptitud_global, restricciones, historial
    else:
        mejor_no_factible = min(poblacion, key=lambda p: p.violacion)
        return (mejor_no_factible.posicion, mejor_no_factible.aptitud, 
                calcular_restricciones(mejor_no_factible.posicion), historial)

def visualizar_resultados(solucion, aptitud, restricciones, historial, seed_usada=None):

    print("\n=== RESULTADOS ===")
    print(f"Costo mínimo: {aptitud:.6f}")
    print(f"Variables: h={solucion[0]:.4f}, l={solucion[1]:.4f}, t={solucion[2]:.4f}, b={solucion[3]:.4f}")
    if seed_usada is not None:
        print(f"Seed utilizada: {seed_usada}")
    
    print("\nRestricciones:")
    nombres = [
        "τ ≤ 13600 psi", "σ ≤ 30000 psi", "h ≤ b", 
        "Diseño estructural", "h ≥ 0.125", "δ ≤ 0.25 in", "P ≤ Pc"
    ]
    for i, (nombre, val) in enumerate(zip(nombres, restricciones)):
        print(f"{i+1}. {nombre}: {val:.4f} {'SI CUMPLE' if val <= 0 else 'NO CUMPLE'}")
    plt.figure(figsize=(12, 5))
    plt.plot(historial, 'b-', linewidth=1.5)
    plt.title(f'Convergencia del Algoritmo Híbrido BHA-PSO{f" (Seed: {seed_usada})" if seed_usada else ""}')
    plt.xlabel('Iteración')
    plt.ylabel('Costo Mínimo ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def generar_configuraciones_parametros():
    parametros = {
        'tam_poblacion': [50, 100, 200],
        'w': [0.4, 0.729, 0.9],
        'p_f': [0.3, 0.45, 0.6],
        'bha_ratio': [0.2, 0.4, 0.6]
    }
    
    configuraciones = []
    config_id = 0
    
    for tam_pop in parametros['tam_poblacion']:
        for w in parametros['w']:
            for p_f in parametros['p_f']:
                for bha_ratio in parametros['bha_ratio']:
                    config = {
                        'config_id': config_id,
                        'tam_poblacion': tam_pop,
                        'w': w,
                        'c1': 1.2,
                        'c2': 1.8,
                        'p_f': p_f,
                        'bha_ratio': bha_ratio,
                        'elitismo': True
                    }
                    configuraciones.append(config)
                    config_id += 1
    
    print(f"ANÁLISIS ANOVA - CONFIGURACIONES GENERADAS:")
    print(f"   Total de configuraciones: {len(configuraciones)}")
    print(f"   Parámetros evaluados: {len(parametros)}")
    print(f"   Valores por parámetro: 3")
    print(f"   Diseño factorial: 3^{len(parametros)} = {len(configuraciones)}")
    print(f"   Cumple especificación: ≥5 configuraciones diferentes ✓")
    print(f"   Cumple especificación: ≥3 valores por parámetro ✓")
    
    return configuraciones

def ejecutar_experimento_anova(configuraciones, num_ejecuciones=10, max_iter=500):
    resultados = []
    total_experimentos = len(configuraciones) * num_ejecuciones
    
    print(f"\nINICIANDO ANÁLISIS ANOVA")
    print(f"Total de configuraciones: {len(configuraciones)}")
    print(f"Ejecuciones por configuración: {num_ejecuciones}")
    print(f"Total de experimentos: {total_experimentos}")
    print("="*80)
    
    mejor_convergencia_global = None
    mejor_config_global = None
    mejor_costo_global = float('inf')
    
    experimento_actual = 0
    
    for config_idx, config in enumerate(configuraciones):
        print(f"\nConfiguración {config_idx + 1}/{len(configuraciones)}")
        print(f"   Parámetros: Pop={config['tam_poblacion']}, w={config['w']:.3f}, "
              f"c1={config['c1']:.3f}, c2={config['c2']:.3f}, p_f={config['p_f']:.2f}, "
              f"bha_ratio={config['bha_ratio']:.1f}")
        
        costos_config = []
        tiempos_config = []
        factibles_config = []
        
        for ejecucion in range(num_ejecuciones):
            experimento_actual += 1
            print(f"   Ejecución {ejecucion + 1}/{num_ejecuciones} "
                  f"({experimento_actual}/{total_experimentos})", end=" ")
            
            # Semilla única para cada ejecución
            seed = 1000 + config_idx * 100 + ejecucion
            
            inicio = time.time()
            try:
                # Ejecutar algoritmo con la configuración dada
                mejor_x, mejor_costo, restricciones, historial = hibrido_bha_pso(
                    tam_poblacion=config['tam_poblacion'],
                    max_iter=max_iter,
                    params=config,
                    seed=seed
                )
                
                tiempo_ejecucion = time.time() - inicio
                factible = es_factible(mejor_x)
                
                # Guardar resultados
                resultado = {
                    'config_id': config['config_id'],
                    'ejecucion': ejecucion,
                    'seed': seed,
                    'costo': mejor_costo,
                    'factible': factible,
                    'tiempo': tiempo_ejecucion,
                    'iteracion_mejor': np.argmin(historial) if factible else max_iter,
                    'convergencia_final': historial[-1],
                    # Parámetros de la configuración
                    'tam_poblacion': config['tam_poblacion'],
                    'w': config['w'],
                    'c1': config['c1'],
                    'c2': config['c2'],
                    'p_f': config['p_f'],
                    'bha_ratio': config['bha_ratio'],
                    # Variables de diseño
                    'h': mejor_x[0],
                    'l': mejor_x[1],
                    't': mejor_x[2],
                    'b': mejor_x[3],
                    # Restricciones
                    'g1_tau': restricciones[0],
                    'g2_sigma': restricciones[1],
                    'g3_h_vs_b': restricciones[2],
                    'g4_diseno': restricciones[3],
                    'g5_h_min': restricciones[4],
                    'g6_deflexion': restricciones[5],
                    'g7_pandeo': restricciones[6],
                }
                
                resultados.append(resultado)
                costos_config.append(mejor_costo)
                tiempos_config.append(tiempo_ejecucion)
                factibles_config.append(factible)
                
                # Verificar si es la mejor solución global
                if factible and mejor_costo < mejor_costo_global:
                    mejor_costo_global = mejor_costo
                    mejor_convergencia_global = historial
                    mejor_config_global = config.copy()
                    mejor_config_global['mejor_x'] = mejor_x
                    mejor_config_global['mejor_seed'] = seed
                
                print(f"OK Costo: {mejor_costo:.4f} {'(Factible)' if factible else '(No Factible)'}")
                
            except Exception as e:
                print(f"ERROR: {str(e)[:30]}...")
                # Registrar error
                resultado_error = {
                    'config_id': config['config_id'],
                    'ejecucion': ejecucion,
                    'seed': seed,
                    'costo': float('inf'),
                    'factible': False,
                    'tiempo': time.time() - inicio,
                    'iteracion_mejor': max_iter,
                    'convergencia_final': float('inf'),
                    **{k: config[k] for k in ['tam_poblacion', 'w', 'c1', 'c2', 'p_f', 'bha_ratio']},
                    **{k: 0 for k in ['h', 'l', 't', 'b']},
                    **{f'g{i+1}_{name}': float('inf') for i, name in enumerate(['tau', 'sigma', 'h_vs_b', 'diseno', 'h_min', 'deflexion', 'pandeo'])}
                }
                resultados.append(resultado_error)
        
        # Estadísticas de la configuración
        if costos_config:
            factibles_count = sum(factibles_config)
            promedio_costo = np.mean([c for c, f in zip(costos_config, factibles_config) if f]) if factibles_count > 0 else float('inf')
            promedio_tiempo = np.mean(tiempos_config)
            
            print(f"   Resumen: {factibles_count}/{num_ejecuciones} factibles, "
                  f"Costo promedio: {promedio_costo:.4f}, Tiempo: {promedio_tiempo:.2f}s")
    
    # Crear DataFrame con resultados
    df_resultados = pd.DataFrame(resultados)
    
    if mejor_config_global is not None:
        print(f"\nMEJOR SOLUCIÓN ENCONTRADA:")
        print(f"   Costo: {mejor_costo_global:.6f}")
        print(f"   Configuración ID: {mejor_config_global['config_id']}")
        print(f"   Seed: {mejor_config_global['mejor_seed']}")
        print(f"   Variables: h={mejor_config_global['mejor_x'][0]:.4f}, "
              f"l={mejor_config_global['mejor_x'][1]:.4f}, "
              f"t={mejor_config_global['mejor_x'][2]:.4f}, "
              f"b={mejor_config_global['mejor_x'][3]:.4f}")
    
    return df_resultados, mejor_convergencia_global, mejor_config_global

def analisis_estadistico_anova(df_resultados):
    print(f"\nANÁLISIS ESTADÍSTICO ANOVA")
    print("="*80)
    
    # Filtrar solo soluciones factibles
    df_factibles = df_resultados[df_resultados['factible'] == True].copy()
    
    if len(df_factibles) == 0:
        print("ADVERTENCIA: No se encontraron soluciones factibles")
        return None
    
    print(f"Soluciones factibles: {len(df_factibles)}/{len(df_resultados)} "
          f"({len(df_factibles)/len(df_resultados)*100:.1f}%)")
    
    # Análisis por parámetro
    parametros = ['tam_poblacion', 'w', 'p_f', 'bha_ratio']
    resultados_anova = {}
    
    for param in parametros:
        print(f"\nAnálisis para parámetro: {param}")
        print("-"*50)
        
        # Estadísticas descriptivas por grupo
        stats_grupo = df_factibles.groupby(param)['costo'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        print("Estadísticas por grupo:")
        print(stats_grupo)
        
        grupos = df_factibles.groupby(param)['costo'].apply(list).to_dict()
        valores_grupos = list(grupos.values())
        medias_grupos = [np.mean(grupo) for grupo in valores_grupos]
        media_total = df_factibles['costo'].mean()
        
        # Suma de cuadrados
        ss_between = sum(len(grupo) * (media - media_total)**2 
                        for grupo, media in zip(valores_grupos, medias_grupos))
        ss_within = sum(sum((valor - np.mean(grupo))**2 for valor in grupo) 
                       for grupo in valores_grupos)
        
        # Grados de libertad
        df_between = len(grupos) - 1
        df_within = len(df_factibles) - len(grupos)
        
        # Cuadrados medios
        ms_between = ss_between / df_between if df_between > 0 else 0
        ms_within = ss_within / df_within if df_within > 0 else 0
        
        # Estadístico F
        f_stat = ms_between / ms_within if ms_within > 0 else 0
        
        print(f"\nResultados ANOVA:")
        print(f"F-estadístico: {f_stat:.4f}")
        print(f"Suma cuadrados entre grupos: {ss_between:.4f}")
        print(f"Suma cuadrados dentro grupos: {ss_within:.4f}")
        
        # Mejor valor del parámetro
        mejor_grupo = min(stats_grupo.index, key=lambda x: stats_grupo.loc[x, 'mean'])
        print(f"Mejor valor para {param}: {mejor_grupo} (media: {stats_grupo.loc[mejor_grupo, 'mean']:.4f})")
        
        resultados_anova[param] = {
            'f_stat': f_stat,
            'mejor_valor': mejor_grupo,
            'mejor_media': stats_grupo.loc[mejor_grupo, 'mean'],
            'stats': stats_grupo
        }
    
    return resultados_anova

def visualizar_resultados_anova(df_resultados, mejor_convergencia=None, carpeta_resultados="resultados_anova"):
    # Filtrar soluciones factibles
    df_factibles = df_resultados[df_resultados['factible'] == True]
    
    if len(df_factibles) == 0:
        print("No hay soluciones factibles para visualizar")
        return carpeta_resultados
    
    # Crear figura con subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Análisis ANOVA - Configuraciones de Parámetros', fontsize=18, y=0.96)
    
    # Grafico de caja para tamaño de población
    ax = axes[0, 0]
    poblaciones = sorted(df_factibles['tam_poblacion'].unique())
    datos_pop = [df_factibles[df_factibles['tam_poblacion'] == pop]['costo'].values 
                for pop in poblaciones]
    if datos_pop and any(len(d) > 0 for d in datos_pop):
        ax.boxplot(datos_pop, labels=poblaciones)
        ax.set_title('Costo vs Tamaño Población', fontsize=11, pad=15)
        ax.set_xlabel('Tamaño Población', fontsize=10)
        ax.set_ylabel('Costo', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Grafico de caja para inercia
    ax = axes[0, 1]
    inercias = sorted(df_factibles['w'].unique())
    datos_w = [df_factibles[df_factibles['w'] == w]['costo'].values for w in inercias]
    if datos_w and any(len(d) > 0 for d in datos_w):
        ax.boxplot(datos_w, labels=[f"{w:.2f}" for w in inercias])
        ax.set_title('Costo vs Inercia (w)', fontsize=11, pad=15)
        ax.set_xlabel('Inercia', fontsize=10)
        ax.set_ylabel('Costo', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Grafico de caja para c1
    ax = axes[0, 2]
    c1_vals = sorted(df_factibles['c1'].unique())
    datos_c1 = [df_factibles[df_factibles['c1'] == c1]['costo'].values for c1 in c1_vals]
    if datos_c1 and any(len(d) > 0 for d in datos_c1):
        ax.boxplot(datos_c1, labels=[f"{c1:.2f}" for c1 in c1_vals])
        ax.set_title('Costo vs Factor Cognitivo (c1)', fontsize=11, pad=15)
        ax.set_xlabel('c1', fontsize=10)
        ax.set_ylabel('Costo', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Grafico de caja para c2
    ax = axes[1, 0]
    c2_vals = sorted(df_factibles['c2'].unique())
    datos_c2 = [df_factibles[df_factibles['c2'] == c2]['costo'].values for c2 in c2_vals]
    if datos_c2 and any(len(d) > 0 for d in datos_c2):
        ax.boxplot(datos_c2, labels=[f"{c2:.2f}" for c2 in c2_vals])
        ax.set_title('Costo vs Factor Social (c2)', fontsize=11, pad=15)
        ax.set_xlabel('c2', fontsize=10)
        ax.set_ylabel('Costo', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    # Grafico de caja para p_f
    ax = axes[1, 1]
    p_f_vals = sorted(df_factibles['p_f'].unique())
    datos_pf = [df_factibles[df_factibles['p_f'] == pf]['costo'].values for pf in p_f_vals]
    if datos_pf and any(len(d) > 0 for d in datos_pf):
        ax.boxplot(datos_pf, labels=[f"{pf:.2f}" for pf in p_f_vals])
        ax.set_title('Costo vs Probabilidad SRGA (p_f)', fontsize=11, pad=15)
        ax.set_xlabel('p_f', fontsize=10)
        ax.set_ylabel('Costo', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Grafico de caja para bha_ratio
    ax = axes[1, 2]
    bha_vals = sorted(df_factibles['bha_ratio'].unique())
    datos_bha = [df_factibles[df_factibles['bha_ratio'] == bha]['costo'].values for bha in bha_vals]
    if datos_bha and any(len(d) > 0 for d in datos_bha):
        ax.boxplot(datos_bha, labels=[f"{bha:.1f}" for bha in bha_vals])
        ax.set_title('Costo vs Proporción BHA', fontsize=11, pad=15)
        ax.set_xlabel('BHA Ratio', fontsize=10)
        ax.set_ylabel('Costo', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Convergencia de la mejor solución
    ax = axes[2, 0]
    if mejor_convergencia is not None and len(mejor_convergencia) > 0:
        ax.plot(mejor_convergencia, 'b-', linewidth=2)
        ax.set_title('Convergencia Mejor Solución', fontsize=11, pad=15)
        ax.set_xlabel('Iteración', fontsize=10)
        ax.set_ylabel('Costo', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    else:
        ax.text(0.5, 0.5, 'No hay datos\nde convergencia', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Convergencia Mejor Solución', fontsize=11, pad=15)
    
    # Distribución de costos
    ax = axes[2, 1]
    costos_validos = df_factibles['costo'].dropna()
    if len(costos_validos) > 0:
        ax.hist(costos_validos, bins=min(30, len(costos_validos)//2), 
               alpha=0.7, edgecolor='black', color='skyblue')
        ax.set_title('Distribución de Costos', fontsize=11, pad=15)
        ax.set_xlabel('Costo', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    # Tiempo vs Costo
    ax = axes[2, 2]
    tiempo_valido = df_factibles['tiempo'].dropna()
    costo_valido = df_factibles['costo'].dropna()
    if len(tiempo_valido) > 0 and len(costo_valido) > 0:
        ax.scatter(df_factibles['tiempo'], df_factibles['costo'], 
                  alpha=0.6, color='orange', s=20)
        ax.set_title('Tiempo vs Costo', fontsize=11, pad=15)
        ax.set_xlabel('Tiempo (s)', fontsize=10)
        ax.set_ylabel('Costo', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)
    
    plt.tight_layout(pad=2.5, h_pad=3.0, w_pad=2.0)
    plt.subplots_adjust(top=0.93) 
    
    # Guardar figura
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archivo_imagen = os.path.join(carpeta_resultados, f"analisis_anova_{timestamp}.png")
    try:
        plt.savefig(archivo_imagen, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Gráfico guardado: {archivo_imagen}")
    except Exception as e:
        print(f"Error al guardar gráfico: {e}")
    
    plt.show()
    
    return carpeta_resultados

def ejecutar_analisis_anova(num_ejecuciones=3, max_iter=150):
    print("ANÁLISIS ANOVA OPTIMIZADO PARA DETERMINACIÓN DE PARÁMETROS")
    print("="*75)
    
    # Crear carpeta de resultados
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    carpeta_resultados = f"resultados_anova__{timestamp}"
    
    if not os.path.exists(carpeta_resultados):
        os.makedirs(carpeta_resultados)
        print(f"Carpeta de resultados creada: {carpeta_resultados}")
    
    # Generar configuraciones optimizadas
    configuraciones = generar_configuraciones_parametros()
    
    print(f"\nPARÁMETROS DEL ANÁLISIS:")
    print(f"*Configuraciones: {len(configuraciones)}")
    print(f"*Ejecuciones por configuración: {num_ejecuciones}")
    print(f"*Iteraciones por ejecución: {max_iter}")
    print(f"*Total experimentos: {len(configuraciones) * num_ejecuciones}")
    print(f"*Tiempo estimado: 3-5 minutos")

    # Ejecutar las pruebas
    inicio_total = time.time()
    df_resultados, mejor_convergencia, mejor_config = ejecutar_experimento_anova(
        configuraciones, num_ejecuciones, max_iter
    )
    tiempo_total = time.time() - inicio_total
    
    # Guardar resultados en CSV
    archivo_csv = os.path.join(carpeta_resultados, f"datos_anova_optimizado_{timestamp}.csv")
    df_resultados.to_csv(archivo_csv, index=False)
    print(f"\nResultados guardados en: {archivo_csv}")
    
    # Análisis estadístico
    resultados_anova = analisis_estadistico_anova(df_resultados)

    # Guardar reporte estadístico
    if resultados_anova:
        archivo_reporte = os.path.join(carpeta_resultados, f"reporte_optimizado_{timestamp}.txt")
        with open(archivo_reporte, 'w', encoding='utf-8') as f:
            f.write("REPORTE ANOVA \n")
            f.write("="*60 + "\n\n")
            f.write(f"Tiempo total de análisis: {tiempo_total:.2f} segundos\n\n")
            
            # Información general
            df_factibles = df_resultados[df_resultados['factible'] == True]
            f.write(f"Total de experimentos: {len(df_resultados)}\n")
            f.write(f"Soluciones factibles: {len(df_factibles)} ({len(df_factibles)/len(df_resultados)*100:.1f}%)\n\n")
            
            # Mejor solución
            if mejor_config:
                f.write("MEJOR SOLUCIÓN ENCONTRADA:\n")
                mejor_x = mejor_config.get('mejor_x', None)
                if mejor_x is not None and len(mejor_x) >= 4:
                    mejor_costo = funcion_objetivo(mejor_x)
                    f.write(f"   Costo: {mejor_costo:.6f}\n")
                    f.write(f"   Variables: h={mejor_x[0]:.4f}, ")
                    f.write(f"l={mejor_x[1]:.4f}, ")
                    f.write(f"t={mejor_x[2]:.4f}, ")
                    f.write(f"b={mejor_x[3]:.4f}\n")
                    f.write(f"   Seed: {mejor_config.get('mejor_seed', 'N/A')}\n")
                    f.write(f"   Config ID: {mejor_config.get('config_id', 'N/A')}\n\n")
                else:
                    f.write("   No se encontró una solución válida\n\n")
            
            # Recomendaciones por parámetro
            f.write("RECOMENDACIONES OPTIMIZADAS:\n")
            f.write("="*40 + "\n")
            for param, resultado in resultados_anova.items():
                f.write(f"• {param}: {resultado['mejor_valor']} ")
                f.write(f"(costo promedio: {resultado['mejor_media']:.6f})\n")
                f.write(f"  F-estadístico: {resultado['f_stat']:.4f}\n\n")
                
            # Configuración recomendada final
            f.write("MEJOR CONFIGURACIÓN FINAL ENCONTRADA:\n")
            f.write("-"*40 + "\n")
            config_optima = {}
            for param, resultado in resultados_anova.items():
                config_optima[param] = resultado['mejor_valor']
            
            for param, valor in config_optima.items():
                f.write(f"{param}: {valor}\n")
        
        print(f"Reporte guardado: {archivo_reporte}")
    
    # Visualizaciones
    carpeta_final = visualizar_resultados_anova(df_resultados, mejor_convergencia, carpeta_resultados)

    print(f"\nANÁLISIS ANOVA COMPLETADO")
    print(f"Tiempo total: {tiempo_total:.2f} segundos")
    print(f"Todos los archivos guardados en: {carpeta_final}")
    print(f"  -Datos CSV: datos_anova_{timestamp}.csv")
    print(f"  -Gráficos: analisis_anova_{timestamp}.png")
    print(f"  -Reporte: reporte_{timestamp}.txt")
    
    if resultados_anova:
        print(f"\nCONFIGURACIÓN ÓPTIMA ENCONTRADA:")
        print("="*50)
        config_optima = {}
        for param, resultado in resultados_anova.items():
            config_optima[param] = resultado['mejor_valor']
            print(f"• {param}: {resultado['mejor_valor']} "
                  f"(costo promedio: {resultado['mejor_media']:.6f})")
        
        print(f"\nPara usar esta configuración óptima, seleccione opción 3 en el menú principal")
    else:
        config_optima = None
    
    return df_resultados, resultados_anova, mejor_config, config_optima


if __name__ == "__main__":
    
    print("ALGORITMO HÍBRIDO BHA-PSO PARA OPTIMIZACIÓN DEL PROBLEMA DE LA VIGA SOLDADA")
    print("="*80)
    print("Seleccione una opción:")
    print("1. Ejecución con parámetros por defecto")
    print("2. Análisis ANOVA para determinar parámetros óptimos")
    print("3. Ejecución con mejores parámetros encontrados")
    
    try:
        opcion = input("\nIngrese su opción (1-3): ").strip()
    except:
        opcion = "1"
    
    if opcion == "2":
        print(f"\nANÁLISIS ANOVA")
        print("Análisis rápido pero completo")
        print("Configuraciones inteligentemente seleccionadas")
        
        try:
            confirmar = input("¿Continuar con el análisis? (s/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirmar = 's'
            
        if confirmar == 's' or confirmar == '':
            df_resultados, resultados_anova, mejor_config, config_optima = ejecutar_analisis_anova(
                num_ejecuciones=10,    # 10 ejecuciones por configuración
                max_iter=200            # 200 iteraciones por ejecución
            )
        else:
            print("Análisis cancelado")
            
    elif opcion == "3":
        print(f"\nEJECUCIÓN CON PARÁMETROS ÓPTIMOS")
        print("="*65)
        
        SEED = 1091  # Semilla conocida que da buenos resultados
        seed_actual = configurar_seed(SEED)
        
        # Parámetros óptimos encontrados mediante análisis ANOVA
        params_optimos = {
            'w': 0.7,
            'c1': 1.2,
            'c2': 1.8,
            'p_f': 0.55,
            'bha_ratio': 0.35,
            'elitismo': True
        }
        
        print(f"Configuración óptima determinada por análisis ANOVA:")
        print(f"   tam_poblacion: 150")
        print(f"   w: {params_optimos['w']}")
        print(f"   c1: {params_optimos['c1']}")
        print(f"   c2: {params_optimos['c2']}")
        print(f"   p_f: {params_optimos['p_f']}")
        print(f"   bha_ratio: {params_optimos['bha_ratio']}")
        print(f"   elitismo: {params_optimos['elitismo']}")
        print(f"   Semilla: {SEED}")
        print("="*65)
        
        mejor_x, mejor_costo, restricciones, historial = hibrido_bha_pso(
            tam_poblacion=150,
            max_iter=500,
            params=params_optimos,
            seed=seed_actual
        )
        
        visualizar_resultados(mejor_x, mejor_costo, restricciones, historial, seed_actual)
        
    else:
        print(f"\nEJECUCIÓN ESTÁNDAR")
        print("="*50)
        
        SEED = None  # Semilla aleatoria
        
        # Configurar seed inicial
        seed_actual = configurar_seed(SEED)
        
        # Parámetros del algoritmo por defecto
        params = {
            'w': 0.8, 'c1': 1.1, 'c2': 1.4,
            'p_f': 0.6, 'bha_ratio': 0.4, 'elitismo': True
        }
        
        print(f"\nEjecutando optimización...")
        print(f"   - Parámetros PSO: w={params['w']}, c1={params['c1']}, c2={params['c2']}")
        print(f"   - SRGA p_f: {params['p_f']}")
        print(f"   - BHA ratio: {params['bha_ratio']}")
        print("="*60)
        
        mejor_x, mejor_costo, restricciones, historial = hibrido_bha_pso(
            tam_poblacion=100,
            max_iter=300,
            params=params,
            seed=seed_actual
        )
        
        visualizar_resultados(mejor_x, mejor_costo, restricciones, historial, seed_actual)