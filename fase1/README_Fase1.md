# Proyecto Centinela · Fase 1 (Módulo 1)
## Centinela del Río — Alerta temprana de calidad del agua de consumo

**Universidad Santo Tomás · Maestría en Ciencia de Datos**
Profundización I: Redes Neuronales — Deep Learning (EA-N-F-004)

**Autores:** Camilo Chiquiza · Brian Lerma · Luz Villarraga
**Modalidad:** individual/grupo · **Framework:** PyTorch

---

## ¿Qué es esto?

La **línea base** de un sistema de alerta temprana que predice, con **un día de
anticipación**, si el agua de una fuente de abastecimiento quedará **fuera del rango
de pH seguro para consumo humano** (6,5–8,5 unidades, guías OMS). Es un problema de
**clasificación binaria** resuelto con un **Perceptrón Multicapa (MLP)** entrenado
desde cero sobre características tabulares.

**Cliente:** junta administradora de un acueducto comunitario rural (JAAR).

## Datos

- **Fuente:** *Water Quality Prediction* (id 733), UCI Machine Learning Repository.
- **Licencia:** Creative Commons Attribution 4.0 (CC BY 4.0).
- **Cita:** Zhao, L. (2019). *Water Quality Prediction* [Conjunto de datos]. UCI
  Machine Learning Repository. https://doi.org/10.1145/3339823
- **Formato:** `.mat` (MATLAB). Se carga con `scipy.io.loadmat` (no es importable con
  `ucimlrepo`). Estructura: serie de 37 estaciones × 705 días, 11 índices
  físico-químicos. Valores normalizados (pH ÷10).

## Cómo ejecutar (Google Colab)

1. Sube `notebooks/Centinela_Fase1_MLP.ipynb` a Colab.
2. Sube `data/water_dataset.mat` a la sesión (la celda 0 lo pide automáticamente).
3. *Entorno de ejecución → Ejecutar todo.*

## Modelo (nivel estratégico)

MLP de **3 capas ocultas** (64-32-16) con:
- **BatchNorm** + ReLU + Dropout
- **`pos_weight`** en la pérdida → reduce falsos negativos (el error costoso)
- **weight decay** (L2) + **scheduler** (ReduceLROnPlateau)
- **early stopping** con restauración de la mejor época
- semilla fija (reproducibilidad)

Salida = **logit** + `BCEWithLogitsLoss` (evita la doble sigmoide).

## Resultados (conjunto de prueba)

| Métrica | Valor |
|---|---|
| Exactitud | 0,874 |
| ROC-AUC | 0,955 |
| PR-AUC (AP) | 0,929 |
| Recall (Riesgo) | 0,892 |
| Falsos negativos | 435 |

## Visualización de datos

Las gráficas del notebook aplican principios de comunicación visual (Cairo, Few,
Nussbaumer Knaflic, Munzner): títulos que comunican el hallazgo (no el tipo de
gráfico), alto *data-ink* sin *chartjunk*, **paleta accesible para daltonismo**
(Okabe-Ito: azul/ámbar en vez de rojo/verde), y anotaciones que dirigen la atención
al punto clave. Se distingue entre gráficas *exploratorias* (equipo técnico) y
*explicativas* (cliente JAAR).

## Hoja de ruta del proyecto

| Fase | Qué se construye |
|---|---|
| **1 (esta)** | MLP tabular desde cero (línea base). |
| 2 | Multimodal: CNN (imagen) + RNN/LSTM/GRU (serie) + fusión. |
| 3 | Industrialización: GPU, precisión mixta, despliegue *offline* en campo. |

## Visualización (marco del curso)

Las gráficas distinguen **exploración** (equipo técnico: correlaciones, ROC, PR,
matriz de confusión) de **explicación** (cliente: "¿el Centinela avisó?"). Se aplican
jerarquía visual y atributos preatentivos (color solo en el foco, gris para el
contexto), control de carga cognitiva y ética visual (ejes no truncados, sin comparar
escalas distintas). El informe incluye un **log auditable de decisiones de
visualización**. Referentes: Cairo (2019); Few (2012); Nussbaumer Knaflic (2015);
Munzner (2014).

## Estructura del repositorio

```
.
├── README.md
├── data/
│   └── water_dataset.mat
├── notebooks/
│   └── Centinela_Fase1_MLP.ipynb
├── docs/
│   ├── Propuesta_Datos_Centinela_Fase1.docx
│   ├── Informe_Tecnico_Etico_Centinela_Fase1.pdf
│   └── Bitacora_IA_Centinela_Fase1.docx
└── figures/
    ├── fig_explicativa.png   (para el cliente)
    ├── fig_aprende.png       (para el cliente)
    ├── fig_roc_pr.png        (exploratoria)
    └── fig_matriz.png        (exploratoria)
```

## Ética

El **falso negativo** (no avisar habiendo riesgo) es el error más costoso: implica
consumo de agua insegura sin prevención. El modelo prioriza reducirlo (`pos_weight`,
umbral ajustable), conforme al **Marco Ético para la IA en Colombia** (Minciencias,
2021). Limitación: datos de Georgia (EE. UU.) → requiere validación local en Colombia.
El sistema es **ayuda a la decisión**, no sustituye el muestreo de laboratorio.
