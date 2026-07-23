# Proyecto Centinela · Fase 2 (Módulo 2)
## Centinela del Río — Sistema multimodal de alerta temprana (CNN + GRU + Fusión)

**Universidad Santo Tomás · Maestría en Ciencia de Datos**
Profundización I: Redes Neuronales — Deep Learning (EA-N-F-004)

**Autores:** Camilo Chiquiza · Brian Lerma · Luz Villarraga
**Framework:** PyTorch · **Entorno:** Google Colab

---

## ¿Qué es esto?

La Fase 2 transforma la línea base tabular de la Fase 1 (MLP) en un **sistema
multimodal especializado**: una rama de visión por computador con *Transfer
Learning* (Rama A), una rama recurrente para la serie físico-química (Rama B)
y una **fusión tardía de embeddings** que integra ambas (Rama C).

**Cliente:** junta administradora de un acueducto comunitario rural (JAAR).

## Estructura de la entrega

```
├── notebooks/
│   ├── Centinela_Fase2_Rama_A.ipynb              # CNN: ResNet50 vs EfficientNet-B0 + Grad-CAM
│   ├── Centinela_Fase2_Rama_B.ipynb              # RNN vs LSTM vs GRU (ventana 14 días)
│   ├── Centinela_Fase2_Rama_B_Desvanecimiento.ipynb  # Anexo: evidencia del desvanecimiento (ejecutado)
│   └── Centinela_Fase2_Rama_C_CORREGIDA.ipynb    # Fusión multimodal con alineación declarada
├── informe/
│   ├── Informe_Tecnico_Centinela_Fase2.pdf       # Máx. 8 páginas (5 pág.)
│   └── Informe_Tecnico_Centinela_Fase2.docx
├── bitacora/
│   ├── Bitacora_IA_Centinela_Fase2.pdf           # 8 entradas A/M/R + auditoría (entrada 5)
│   └── Bitacora_IA_Centinela_Fase2.docx
├── figuras/                                       # Visualizaciones estáticas comentadas
│   ├── cliente_resumen_Fase2.png                 # Para el cliente (lenguaje no técnico)
│   ├── matrices_confusion_RamaA.png · gradcam_ResNet50_RamaA.png
│   ├── desvanecimiento_tarea_RamaB.png · desvanecimiento_normas_RamaB.png
│   └── (agregar ablacion_RamaC.png generada al ejecutar la Rama C)
├── pesos/
│   ├── mejor_clf_ResNet50.pth · mejor_clf_EfficientNet_B0.pth   # Rama A
│   ├── mejor_RNN.pth · mejor_LSTM.pth · mejor_GRU.pth           # Rama B
│   └── (agregar Centinela_Fase2_RamaC_Fusion.pth de la ejecución de la Rama C)
└── data/  → water_dataset.mat + Imagenes_agua/train/ (512 imágenes, 3 clases)
```

## Datos (política de la actividad)

| Rama | Fuente | Licencia | Cumplimiento |
|---|---|---|---|
| B (serie) | Zhao, L. (2019). *Water Quality Prediction* (id 733). UCI ML Repository. https://doi.org/10.1145/3339823 | CC BY 4.0 | 26K observaciones ≥ 1.000 |
| A (imagen) | *Water Classification Dataset*. Roboflow Universe. | CC BY 4.0 | 512 imágenes ≥ 500, 3 clases |

Ninguna fuente proviene de Kaggle ni de sus *mirrors*.

## Declaración de alineación (Rama C)

No existe clave genuina por muestra entre las fuentes (la serie no trae imágenes;
las imágenes no traen fecha/GPS). Se declara **alineación simulada por condición
observable**: cada ventana recibe una imagen coherente con el estado del agua en
su último día (pH dentro/fuera de rango en t), **nunca con la etiqueta futura**
(t+1). Imágenes de train/test provienen de particiones disjuntas. Limitación
declarada: el co-registro real en campo se propone para la Fase 3.

## Resultados

| Componente | Métrica principal |
|---|---|
| Rama A — ResNet50 / EfficientNet-B0 | Exactitud 97,4% / 98,7% · ROC-AUC 0,9997 |
| Rama B — GRU (mejor celda) | ROC-AUC 0,9350 · Recall riesgo 76,9% |
| Anexo desvanecimiento (T=100) | RNN 49,8% (azar) vs LSTM 100% / GRU 99,5% |
| Rama C — Ablación (9.879 ventanas) | Solo serie 0,9289 · Solo imagen 0,5022 · **Fusión 0,9289** |

**Hallazgo honesto:** la fusión iguala el ROC-AUC de la mejor rama individual y
mejora precisión (0,82 vs 0,75) y exactitud (86,1% vs 83,9%) — menos falsas
alarmas al mismo poder de detección. Bajo alineación simulada, la modalidad
visual es mayormente redundante con la serie; su valor pleno exige co-registro
real (Fase 3).

## Cómo ejecutar (Colab)

1. Subir `Centinela_Fase2_insumos.zip` (data + imágenes + pesos) a Google Drive.
2. Abrir el notebook y ejecutar el BLOQUE 1: monta Drive, localiza el ZIP,
   descomprime a disco local y verifica cada insumo con ✅/❌.
3. *Entorno de ejecución → Ejecutar todo.*

## Hoja de ruta

| Fase | Qué se construye |
|---|---|
| 1 | MLP tabular desde cero (línea base). ✔ |
| **2 (esta)** | Multimodal: CNN + GRU + fusión con ablación y alineación declarada. ✔ |
| 3 | Industrialización de la rama visual: GPU, precisión mixta, despliegue *offline* + protocolo de co-registro en campo. |
