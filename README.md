# 🌊 Proyecto Centinela — Centinela del Río
## Sistema de alerta temprana de calidad del agua de consumo · Fases 1, 2 y 3

**Universidad Santo Tomás · Maestría en Ciencia de Datos**
Profundización I: Redes Neuronales — Deep Learning (EA-N-F-004)

**Autores:** Camilo Chiquiza · Brian Lerma
**Framework:** PyTorch (+ Keras en Fase 3) · **Entorno:** Google Colab


---

## ¿Qué es esto?

Un sistema de alerta temprana que predice, **con un día de anticipación**, si el
agua de la fuente de abastecimiento de un acueducto comunitario rural (JAAR)
quedará **fuera del rango de pH seguro para consumo humano** (6,5–8,5, guías
OMS). El proyecto creció por fases a lo largo del curso y este repositorio
contiene las tres fases completas.

| Fase | Qué se construye | Estado |
|---|---|---|
| **1** | Línea base tabular: MLP de 3 capas desde cero sobre 11 indicadores físico-químicos. | ✔ `fase1/` |
| **2** | Sistema multimodal: CNN con *Transfer Learning* (imagen) + GRU (serie temporal) + fusión tardía de *embeddings* con estudio de ablación. | ✔ `fase2/` |
| **3** | Industrialización (MLOps): pipeline optimizado, entrenamiento GPU con precisión mixta, cuantización Edge (INT8) y ruta de despliegue *offline* en tablet. | ✔ `fase3/` |

## Estructura del repositorio

```
├── README.md                  ← este archivo
├── data/
│   ├── water_dataset.mat      ← serie UCI (fases 1 y 2)
│   └── Imagenes_agua/train/   ← 512 imágenes en 3 clases (fases 2 y 3)
│
├── fase1/
│   ├── README_Fase1.md
│   ├── docs/         Propuesta de Datos · Informe Técnico-Ético · Bitácora de IA
│   ├── notebooks/    Centinela_Fase1_MLP.ipynb
│   ├── pesos/        centinela_fase1.pt
│   └── figuras/      curvas, ROC/PR, matriz de confusión, explicativa cliente
│
├── fase2/
│   ├── README_Fase2.md
│   ├── notebooks/    Rama A (CNN) · Rama B (RNN/LSTM/GRU) · Rama C (fusión)
│   ├── informe/      Informe Técnico
│   ├── bitacora/     Bitácora de IA
│   ├── figuras/      Grad-CAM, matrices, ablación
│   └── pesos/        ResNet50 · EfficientNet-B0 · RNN · LSTM · GRU
│
└── fase3/
    ├── README_Fase3.md         ← detalle completo de la fase (léelo)
    ├── cifras_informe.json     ← única fuente de verdad de los números
    ├── notebooks/    Centinela_Fase3.ipynb (ejecutado, con salidas)
    ├── docs/         Informe Técnico · Bitácora de IA · AJUSTES_Fase3.md
    ├── figuras/      pipelines, augmentation, FP32/FP16, matriz, Edge
    ├── ajustes/      scripts de corrección y notebook original
    └── modelos/      modelo_int8_estatico.pth (el que se despliega)
                      ⚠ los pesos completos NO están en git → ver  en --https://drive.google.com/drive/folders/1qpi2XzIK-A4srky-j-Ho0m-AuMbnToil?usp=sharing 
```

> **Nota sobre los modelos pesados:** GitHub limita cada archivo a 100 MB. Los
> pesos completos de la Fase 3 (`.keras` de 270 MB, checkpoints de 128 MB, etc.)
> están publicados en Google Drive / GitHub Releases. El detalle y los enlaces
> están en [`fase3/modelos/README_modelos.md`](https://drive.google.com/drive/folders/1qpi2XzIK-A4srky-j-Ho0m-AuMbnToil?usp=sharing).
> El único peso versionado aquí es el modelo cuantizado INT8 estático (~11 MB),
> que es precisamente el artefacto que se despliega en campo.

## Resultados por fase (conjunto de prueba)

**Fase 1 — MLP tabular:** exactitud 0,874 · ROC-AUC 0,955 · recall de riesgo 0,892.

**Fase 2 — Sistema multimodal:**

| Componente | Métrica principal |
|---|---|
| Rama A — ResNet50 / EfficientNet-B0 | Exactitud 97,4% / 98,7% · ROC-AUC 0,9997 |
| Rama B — GRU (mejor celda) | ROC-AUC 0,9350 · Recall riesgo 76,9% |
| Rama C — Fusión (ablación, 9.879 ventanas) | ROC-AUC 0,9289 · Precisión 0,82 · Exactitud 86,1% |

**Fase 3 — Industrialización:** ResNet-18 con precisión mixta FP16, cuantización
estática INT8 para despliegue Edge y exportación verificada a `.pth`, `.onnx` y
`.keras`. Todos los números provienen de `fase3/cifras_informe.json`, generado
por el notebook — ningún valor del informe se escribe a mano.

## Datos y licencias

| Fuente | Uso | Licencia |
|---|---|---|
| Zhao, L. (2019). *Water Quality Prediction* (id 733). UCI ML Repository. https://doi.org/10.1145/3339823 | Fases 1 y 2 (serie) | CC BY 4.0 |
| *Water Classification Dataset*. Roboflow Universe. | Fases 2 y 3 (imágenes) | CC BY 4.0 |

Ninguna fuente proviene de Kaggle ni de sus *mirrors*.

## Cómo ejecutar (Colab)

- **Fase 1:** abrir `fase1/notebooks/Centinela_Fase1_MLP.ipynb`; solo requiere `data/water_dataset.mat`.
- **Fase 2:** ver instrucciones en `fase2/README_Fase2.md` (montaje de insumos desde Drive).
- **Fase 3:** abrir `fase3/notebooks/Centinela_Fase3.ipynb` en Colab con GPU T4,
  ajustar `RUTA_BASE` y *Ejecutar todo* (~15 min). Detalle en `fase3/README_Fase3.md`.

---

*Universidad Santo Tomás · Maestría en Ciencia de Datos · 2026*
