# Algoritmo Híbrido BHA-PSO para Optimización de Viga Soldada

## Descripción del Problema

Este proyecto implementa un algoritmo híbrido que combina **Black Hole Algorithm (BHA)** y **Particle Swarm Optimization (PSO)** para resolver el problema clásico de optimización de diseño de vigas soldadas. El objetivo es minimizar el costo de fabricación mientras se satisfacen las restricciones estructurales y de seguridad.

### Formulación del Problema

**Variables de diseño:**
- `h` (x₁): Espesor de la soldadura [0.1, 2.0] in
- `l` (x₂): Longitud de la soldadura [0.1, 10.0] in  
- `t` (x₃): Altura de la viga [0.1, 10.0] in
- `b` (x₄): Espesor de la viga [0.1, 2.0] in

**Función objetivo:**
```
Minimizar: f(x) = 1.10471·h²·l + 0.04811·t·b·(14.0 + l)
```

**Restricciones:**
1. Esfuerzo cortante: τ ≤ 13,600 psi
2. Esfuerzo normal: σ ≤ 30,000 psi  
3. Restricción geométrica: h ≤ b
4. Restricción de diseño estructural
5. Espesor mínimo: h ≥ 0.125 in
6. Deflexión máxima: δ ≤ 0.25 in
7. Carga crítica de pandeo: P ≤ Pc

## Características del Algoritmo

- **Algoritmo Híbrido**: Combina exploración global (BHA) con explotación local (PSO)
- **Manejo de Restricciones**: Implementa Stochastic Ranking for Genetic Algorithms (SRGA)
- **Análisis ANOVA**: Sistema completo para determinación óptima de parámetros
- **Reproducibilidad**: Control de semillas aleatorias para resultados replicables

## Requisitos del Sistema

### Versión de Python
```
Python 3.8 o superior (Recomendado: Python 3.9+)
```

### Librerías Requeridas

```
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
```

## Instalación

### 1. Clonar o Descargar el Proyecto
```bash
# Crear directorio del proyecto
mkdir algoritmo-bha-pso
cd algoritmo-bha-pso

# Copiar el archivo bha_pso.py al directorio
```

### 2. Crear Entorno Virtual (Recomendado)
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar Dependencias

#### Opción A: Instalación Manual
```bash
pip install numpy>=1.21.0
pip install matplotlib>=3.5.0  
pip install pandas>=1.3.0
```

#### Opción B: Crear requirements.txt
Crear archivo `requirements.txt`:
```
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
```

Luego instalar:
```bash
pip install -r requirements.txt
```

### 4. Verificar Instalación
```bash
python -c "import numpy, matplotlib, pandas; print('Todas las librerías instaladas correctamente')"
```

## Ejecución del Programa

### Ejecutar el Programa Principal
```bash
python bha_pso.py
```

### Opciones del Menú

El programa presenta tres opciones principales:

#### **Opción 1: Ejecución Estándar**
- Parámetros por defecto del algoritmo
- Ejecución rápida (~30-60 segundos)
- Ideal para pruebas iniciales

#### **Opción 2: Análisis ANOVA**
- Análisis estadístico completo de parámetros
- 81 configuraciones × 10 ejecuciones = 810 experimentos
- Tiempo estimado: 20-30 minutos
- Genera archivos de resultados en una carpeta

#### **Opción 3: Ejecución Óptima**
- Usa parámetros optimizados encontrados por ANOVA
- Configuración: tam_poblacion=200, w=0.9, c1=1.2, c2=1.8, p_f=0.45, bha_ratio=0.2
- Semilla fija para reproducibilidad

## Información Técnica

### Algoritmo Híbrido BHA-PSO
1. **Fase BHA (Exploración)**: Primeras iteraciones buscan globalmente
2. **Fase PSO (Explotación)**: Iteraciones finales refinan soluciones
3. **SRGA**: Manejo estocástico de restricciones

### Análisis ANOVA
- **Diseño factorial**: 3⁴ = 81 configuraciones
- **Replicas**: 10 ejecuciones independientes por configuración
- **Estadísticos**: F-test, suma de cuadrados, medias por grupo
- **Visualizaciones**: Boxplots, histogramas, convergencia
