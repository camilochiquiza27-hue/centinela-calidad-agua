# =====================================================================
# PROYECTO CENTINELA — FASE 3 · CELDAS CORREGIDAS
# ---------------------------------------------------------------------
# Cada BLOQUE de este archivo REEMPLAZA la celda indicada del notebook
# original (Centinela_Fase3.ipynb). Copiar y pegar en Colab en el orden
# indicado, sobre el runtime ya ejecutado hasta el BLOQUE 1.
#
# IMPORTANTE: los valores numéricos NO están escritos aquí. Cada bloque
# imprime lo que mide. El informe y la bitácora se llenan DESPUÉS de
# correr, con las cifras reales. Esa es exactamente la disciplina que
# el docente señaló en la Fase 2.
# =====================================================================


# =====================================================================
# BLOQUE 2-BIS — REEMPLAZA LA CELDA 6 (pipeline)
# ---------------------------------------------------------------------
# QUÉ SE CORRIGE
#   (a) El benchmark tf.data comparaba peras con manzanas: el pipeline
#       "sin optimizar" NO tenía .shuffle() y el "optimizado" SÍ, con
#       buffer = dataset completo (512). Ese shuffle obliga a leer las
#       512 imágenes antes de entregar el primer batch, así que la
#       medición capturaba el costo del barajado, no el del cache.
#   (b) El cache se medía en la PRIMERA pasada, cuando aún se está
#       llenando. Ahora se mide en la SEGUNDA época, que es cuando el
#       cache tiene efecto.
#   (c) Se crea una partición train/val/test REAL y reutilizable, que
#       es lo que faltaba y lo que permitió el error del BLOQUE 7.
# =====================================================================

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, models

IMG_SIZE   = 224
BATCH_SIZE = 32
SEMILLA    = 42
torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASES        = ['Bersih', 'Keruh', 'Kotor']
CLASES_ES     = ['Limpia', 'Turbia', 'Contaminada']

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

transform_aug = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),                          # 1
    transforms.RandomHorizontalFlip(p=0.5),                   # 2
    transforms.RandomRotation(degrees=15),                    # 3
    transforms.ColorJitter(brightness=0.2, contrast=0.2,      # 4
                           saturation=0.1, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)), # 5
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ── PARTICIÓN ÚNICA Y ESTRATIFICADA ─────────────────────────────────
# Se calculan los ÍNDICES una sola vez y se reutilizan en todo el
# notebook. Así train / val / test son siempre los mismos conjuntos,
# sin importar qué transform se aplique.
from sklearn.model_selection import train_test_split

dataset_ref = datasets.ImageFolder(root=RUTA_IMG, transform=transform_val)
N_CLASES    = len(dataset_ref.classes)
etiquetas   = np.array([y for _, y in dataset_ref.samples])
idx_todos   = np.arange(len(dataset_ref))

idx_train, idx_temp = train_test_split(
    idx_todos, test_size=0.30, random_state=SEMILLA, stratify=etiquetas)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, random_state=SEMILLA, stratify=etiquetas[idx_temp])

print('=== PARTICIÓN ESTRATIFICADA (índices fijos) ===')
for nombre, idx in [('Train', idx_train), ('Val', idx_val), ('Test', idx_test)]:
    conteo = np.bincount(etiquetas[idx], minlength=N_CLASES)
    print(f'{nombre:<6} n={len(idx):>4} | ' +
          ' | '.join(f'{CLASES_ES[c]}: {conteo[c]}' for c in range(N_CLASES)))

# COMPROBACIÓN EXPLÍCITA: los conjuntos no se solapan.
# (Lección de la Fase 2: conservar la salida que revela el problema.)
assert len(set(idx_train) & set(idx_test)) == 0, 'FUGA: train y test se solapan'
assert len(set(idx_val)   & set(idx_test)) == 0, 'FUGA: val y test se solapan'
print('Comprobación de fuga: OK — train/val/test disjuntos')

dataset_aug_full = datasets.ImageFolder(root=RUTA_IMG, transform=transform_aug)
ds_tr  = Subset(dataset_aug_full, idx_train)   # con augmentation (en vivo)
ds_vl  = Subset(dataset_ref,      idx_val)     # sin augmentation
ds_te  = Subset(dataset_ref,      idx_test)    # sin augmentation

loader_train = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, prefetch_factor=2)
loader_val   = DataLoader(ds_vl, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
loader_test  = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

# ── BENCHMARK PYTORCH: sin vs con optimización ──────────────────────
def medir_tiempo_pt(loader, n_batches=10):
    inicio = time.time()
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(DISPOSITIVO, non_blocking=True)
        if i >= n_batches:
            break
    return (time.time() - inicio) / n_batches

loader_sin = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False)
loader_con = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, prefetch_factor=2)

print('\nMidiendo PyTorch DataLoader...')
t_sin = medir_tiempo_pt(loader_sin)
t_con = medir_tiempo_pt(loader_con)
mejora = (t_sin - t_con) / t_sin * 100
print(f'PyTorch sin optimizar : {t_sin*1000:.1f} ms/batch')
print(f'PyTorch optimizado    : {t_con*1000:.1f} ms/batch  (mejora {mejora:.1f}%)')

# ── BENCHMARK TF.DATA — COMPARACIÓN JUSTA ───────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

rutas_imgs = [dataset_ref.samples[i][0] for i in idx_train]
etiq_tf    = [dataset_ref.samples[i][1] for i in idx_train]

def cargar_imagen_tf(ruta, etiqueta):
    img = tf.io.read_file(ruta)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - tf.constant(IMAGENET_MEAN)) / tf.constant(IMAGENET_STD)
    return img, etiqueta

base = tf.data.Dataset.from_tensor_slices((rutas_imgs, etiq_tf))

# CORRECCIÓN CLAVE: el MISMO .shuffle() en ambos pipelines.
# Lo único que cambia es map paralelo + cache + prefetch.
ds_tf_sin = (base
             .shuffle(buffer_size=len(rutas_imgs), seed=SEMILLA)
             .map(cargar_imagen_tf)          # serial, sin AUTOTUNE
             .batch(BATCH_SIZE))             # sin cache, sin prefetch

ds_tf_con = (base
             .shuffle(buffer_size=len(rutas_imgs), seed=SEMILLA)
             .map(cargar_imagen_tf, num_parallel_calls=AUTOTUNE)
             .cache()
             .batch(BATCH_SIZE)
             .prefetch(AUTOTUNE))

def medir_epoca_tf(ds):
    inicio = time.time()
    n = 0
    for _ in ds:
        n += 1
    return (time.time() - inicio) / max(n, 1)

print('\nMidiendo tf.data (época 1 y época 2)...')
t_tf_sin_e1 = medir_epoca_tf(ds_tf_sin)
t_tf_sin_e2 = medir_epoca_tf(ds_tf_sin)
t_tf_con_e1 = medir_epoca_tf(ds_tf_con)   # llena el cache
t_tf_con_e2 = medir_epoca_tf(ds_tf_con)   # cache caliente ← la comparación válida

mejora_tf_e1 = (t_tf_sin_e1 - t_tf_con_e1) / t_tf_sin_e1 * 100
mejora_tf_e2 = (t_tf_sin_e2 - t_tf_con_e2) / t_tf_sin_e2 * 100

print(f'\n=== TF.DATA — época 1 (cache llenándose) ===')
print(f'  sin optimizar: {t_tf_sin_e1*1000:.1f} ms/batch')
print(f'  optimizado   : {t_tf_con_e1*1000:.1f} ms/batch  ({mejora_tf_e1:+.1f}%)')
print(f'=== TF.DATA — época 2 (cache caliente) ← COMPARACIÓN VÁLIDA ===')
print(f'  sin optimizar: {t_tf_sin_e2*1000:.1f} ms/batch')
print(f'  optimizado   : {t_tf_con_e2*1000:.1f} ms/batch  ({mejora_tf_e2:+.1f}%)')

# ── TABLA + GRÁFICA ─────────────────────────────────────────────────
print(f'\n{"Framework":<26}{"Sin optim.":>13}{"Optimizado":>13}{"Mejora":>10}')
print('-' * 62)
print(f'{"PyTorch DataLoader":<26}{t_sin*1000:>11.1f}ms{t_con*1000:>11.1f}ms{mejora:>9.1f}%')
print(f'{"tf.data (época 1)":<26}{t_tf_sin_e1*1000:>11.1f}ms{t_tf_con_e1*1000:>11.1f}ms{mejora_tf_e1:>9.1f}%')
print(f'{"tf.data (época 2)":<26}{t_tf_sin_e2*1000:>11.1f}ms{t_tf_con_e2*1000:>11.1f}ms{mejora_tf_e2:>9.1f}%')

fig, ax = plt.subplots(figsize=(10, 4.5))
etiqs = ['PyTorch\nsin optim', 'PyTorch\noptimizado',
         'tf.data ép.1\nsin optim', 'tf.data ép.1\noptimizado',
         'tf.data ép.2\nsin optim', 'tf.data ép.2\noptimizado']
vals = [t_sin*1000, t_con*1000,
        t_tf_sin_e1*1000, t_tf_con_e1*1000,
        t_tf_sin_e2*1000, t_tf_con_e2*1000]
cols = ['#C0392B', '#02C39A'] * 3
bars = ax.bar(etiqs, vals, color=cols, alpha=0.85, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.02,
            f'{v:.0f}ms', ha='center', fontweight='bold', fontsize=9)
ax.set_ylabel('Tiempo por batch (ms)')
ax.set_title('Comparativa de pipelines — mismo shuffle en ambos, medido en época 1 y 2',
             fontsize=11, fontweight='bold')
ax.legend(handles=[mpatches.Patch(color='#C0392B', alpha=.85, label='Sin optimización'),
                   mpatches.Patch(color='#02C39A', alpha=.85, label='Con optimización')])
ax.grid(axis='y', alpha=0.3); ax.spines[['top','right']].set_visible(False)
plt.xticks(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, 'comparativa_pipelines.png'), dpi=150)
plt.show()


# =====================================================================
# BLOQUE 3-BIS — NUEVO · A/B DE AUGMENTATION (lo pedía la guía)
# ---------------------------------------------------------------------
# La guía exige: "comparar el accuracy de validación con y sin aumento
# tras diez épocas". No se había hecho. Cuesta ~2 minutos.
# =====================================================================

def crear_resnet18(n_clases):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n_clases))
    return m

def entrenar_corto(loader_tr, epocas=10, usar_amp=True, etiqueta=''):
    """Entrena desde cero y devuelve (mejor_acc_val, vram_pico_MB, seg/época)."""
    torch.manual_seed(SEMILLA)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = crear_resnet18(N_CLASES).to(DISPOSITIVO)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if usar_amp else None
    mejor, tiempos = 0.0, []

    for ep in range(epocas):
        m.train(); t0 = time.time()
        for imgs, y in loader_tr:
            imgs, y = imgs.to(DISPOSITIVO), y.to(DISPOSITIVO)
            opt.zero_grad()
            if usar_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    loss = crit(m(imgs), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = crit(m(imgs), y)
                loss.backward(); opt.step()
        torch.cuda.synchronize(); tiempos.append(time.time() - t0)

        m.eval(); c = t = 0
        with torch.no_grad():
            for imgs, y in loader_val:
                imgs, y = imgs.to(DISPOSITIVO), y.to(DISPOSITIVO)
                c += (m(imgs).argmax(1) == y).sum().item(); t += y.size(0)
        mejor = max(mejor, c / t)

    vram = torch.cuda.max_memory_allocated() / 1e6
    print(f'  {etiqueta:<28} acc_val={mejor*100:5.1f}% | '
          f'VRAM={vram:6.0f}MB | {np.mean(tiempos):.2f}s/época')
    del m, opt; torch.cuda.empty_cache()
    return mejor, vram, float(np.mean(tiempos))

loader_sin_aug = DataLoader(Subset(dataset_ref, idx_train), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2, pin_memory=True)

print('=== A/B DE DATA AUGMENTATION (10 épocas cada uno) ===')
acc_sin_aug, _, _ = entrenar_corto(loader_sin_aug, 10, True, 'SIN augmentation')
acc_con_aug, _, _ = entrenar_corto(loader_train,   10, True, 'CON augmentation')
print(f'\nDiferencia: {(acc_con_aug - acc_sin_aug)*100:+.1f} pp a favor de '
      f'{"CON" if acc_con_aug > acc_sin_aug else "SIN"} augmentation')
print('NOTA: con 512 imágenes y transfer learning el efecto puede ser pequeño')
print('      o incluso negativo. Se reporta lo que salga, no lo que conviene.')


# =====================================================================
# BLOQUE 4-BIS — NUEVO · FP32 vs FP16 **MEDIDO** (no estimado)
# ---------------------------------------------------------------------
# QUÉ SE CORRIGE
#   El notebook original NO midió FP32. Escribía:
#       VRAM_FP32 = VRAM_FP16 * 1.8      ← multiplicación, no medición
#       t_FP32    = t_FP16    * 1.4      ← ídem
#   y el informe reportaba 1084 MB y 8.5 s como si fueran datos.
#   La rúbrica pide el ahorro "documentado". Una época cuesta ~6 s:
#   medirlo de verdad son 60 segundos de cómputo.
# =====================================================================

print('=== FP32 vs FP16 — MEDICIÓN DIRECTA (5 épocas cada uno) ===')
_, vram_fp32_med, t_fp32_med = entrenar_corto(loader_train, 5, False, 'FP32 (sin AMP)')
_, vram_fp16_med, t_fp16_med = entrenar_corto(loader_train, 5, True,  'FP16 (con AMP)')

ahorro_vram = (1 - vram_fp16_med / vram_fp32_med) * 100
acel        = t_fp32_med / t_fp16_med

print(f'\n{"Métrica":<26}{"FP32":>12}{"FP16":>12}{"Cambio":>14}')
print('-' * 64)
print(f'{"VRAM pico (MB)":<26}{vram_fp32_med:>12.0f}{vram_fp16_med:>12.0f}{ahorro_vram:>13.1f}%')
print(f'{"Tiempo/época (s)":<26}{t_fp32_med:>12.2f}{t_fp16_med:>12.2f}{acel:>13.2f}x')
print('\nAmbas cifras son MEDIDAS en esta corrida. Interpretación: el ahorro')
print('de VRAM es menor al 50% porque AMP mantiene los pesos maestros en')
print('FP32; el ahorro se concentra en las activaciones.')

fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4))
fig2.suptitle('FP32 vs FP16 — valores MEDIDOS en Tesla T4', fontsize=12, fontweight='bold')
for a, (tit, vals, uni) in zip(ax2, [
        ('VRAM pico (MB)', [vram_fp32_med, vram_fp16_med], 'MB'),
        ('Tiempo por época (s)', [t_fp32_med, t_fp16_med], 's')]):
    bars = a.bar(['FP32', 'FP16'], vals, color=['#C0392B', '#02C39A'], alpha=0.85)
    for b, v in zip(bars, vals):
        a.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.02,
               f'{v:.1f}{uni}', ha='center', fontweight='bold')
    a.set_title(tit, fontweight='bold'); a.grid(axis='y', alpha=.3)
    a.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, 'comparativa_fp32_fp16.png'), dpi=150)
plt.show()


# =====================================================================
# BLOQUE 4-TER — NUEVO · EVIDENCIA DE CUDA OOM Y SU SOLUCIÓN
# ---------------------------------------------------------------------
# El nivel Estratégico del criterio 2 pide "resuelve CUDA OOM con
# evidencia". Se provoca a propósito y se documenta la mitigación.
# =====================================================================

print('=== GESTIÓN DE VRAM: PROVOCAR Y RESOLVER UN CUDA OOM ===')
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
modelo_oom = crear_resnet18(N_CLASES).to(DISPOSITIVO)
opt_oom    = torch.optim.Adam(modelo_oom.parameters(), lr=1e-4)
crit_oom   = nn.CrossEntropyLoss()

batch_probado, oom_en = None, None
for bs in [256, 512, 1024, 2048, 4096]:
    try:
        x = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE, device=DISPOSITIVO)
        y = torch.randint(0, N_CLASES, (bs,), device=DISPOSITIVO)
        opt_oom.zero_grad()
        crit_oom(modelo_oom(x), y).backward()
        opt_oom.step()
        pico = torch.cuda.max_memory_allocated()/1e6
        print(f'  batch={bs:<5} OK    | VRAM pico: {pico:7.0f} MB')
        batch_probado = bs
        del x, y; torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        oom_en = bs
        print(f'  batch={bs:<5} CUDA OOM  ← {str(e).splitlines()[0][:70]}')
        torch.cuda.empty_cache()
        break

if oom_en:
    print(f'\nOOM reproducido en batch={oom_en}. Mitigaciones aplicables:')
    print(f'  1. Reducir el batch (el proyecto usa {BATCH_SIZE}, muy holgado)')
    print(f'  2. Precisión mixta FP16 → menos memoria en activaciones')
    print(f'  3. Acumulación de gradientes → batch efectivo grande, memoria de uno chico')
    print(f'  4. torch.utils.checkpoint → recomputa activaciones en el backward')
    print(f'\nVerificación de la mitigación 1+2 (batch={oom_en} con AMP):')
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        sc = torch.amp.GradScaler('cuda')
        acum, sub = 4, oom_en // 4
        opt_oom.zero_grad()
        for _ in range(acum):
            x = torch.randn(sub, 3, IMG_SIZE, IMG_SIZE, device=DISPOSITIVO)
            y = torch.randint(0, N_CLASES, (sub,), device=DISPOSITIVO)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = crit_oom(modelo_oom(x), y) / acum
            sc.scale(loss).backward()
            del x, y
        sc.step(opt_oom); sc.update()
        print(f'  RESUELTO: batch efectivo {oom_en} vía {acum} pasos de {sub} + AMP')
        print(f'  VRAM pico: {torch.cuda.max_memory_allocated()/1e6:.0f} MB')
    except torch.cuda.OutOfMemoryError:
        print('  Aún OOM: reducir más el sub-batch.')
else:
    print(f'\nNo hubo OOM hasta batch={batch_probado}. La T4 (15.6 GB) tiene margen')
    print('de sobra para ResNet-18. Se documenta el límite empírico alcanzado.')

del modelo_oom, opt_oom; torch.cuda.empty_cache()


# =====================================================================
# BLOQUE 6-BIS — REEMPLAZA LA CELDA 14 (evaluación) · + INTERVALOS
# ---------------------------------------------------------------------
# QUÉ SE CORRIGE
#   El docente señaló en la Fase 2: "Reporten la incertidumbre. Para el
#   ROC-AUC de la rama visual, den un intervalo de confianza en vez de
#   cuatro decimales sobre 78 imágenes". Se añade bootstrap.
# =====================================================================

from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, classification_report)

def evaluar(m, loader, dispositivo, usar_amp=True):
    m.eval(); yt, yp, ypr = [], [], []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(dispositivo)
            if usar_amp and dispositivo.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = m(imgs)
            else:
                logits = m(imgs)
            logits = logits.float()
            ypr.extend(F.softmax(logits, 1).cpu().numpy())
            yp.extend(logits.argmax(1).cpu().numpy())
            yt.extend(y.numpy())
    return np.array(yt), np.array(yp), np.array(ypr)

def ic_bootstrap(y_true, y_pred, y_prob, n=2000, alfa=0.05):
    """IC percentil por bootstrap para accuracy y ROC-AUC macro-OvR."""
    rng, accs, aucs = np.random.default_rng(SEMILLA), [], []
    N = len(y_true)
    for _ in range(n):
        s = rng.integers(0, N, N)
        if len(np.unique(y_true[s])) < len(np.unique(y_true)):
            continue                      # remuestreo sin todas las clases
        accs.append((y_true[s] == y_pred[s]).mean())
        try:
            aucs.append(roc_auc_score(y_true[s], y_prob[s],
                                      multi_class='ovr', average='macro'))
        except ValueError:
            pass
    q = [100*alfa/2, 100*(1-alfa/2)]
    return np.percentile(accs, q), np.percentile(aucs, q)

y_true, y_pred, y_prob = evaluar(modelo, loader_test, DISPOSITIVO)
acc = (y_true == y_pred).mean()
auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
(ic_acc_lo, ic_acc_hi), (ic_auc_lo, ic_auc_hi) = ic_bootstrap(y_true, y_pred, y_prob)

print(f'=== RESULTADOS EN TEST (n={len(y_true)}) ===')
print(f'Accuracy: {acc*100:.1f}%   IC95% [{ic_acc_lo*100:.1f}% – {ic_acc_hi*100:.1f}%]')
print(f'ROC-AUC : {auc:.3f}      IC95% [{ic_auc_lo:.3f} – {ic_auc_hi:.3f}]')
print(f'\nEl intervalo es ancho porque n={len(y_true)}. Reportar cuatro')
print('decimales sobre esta muestra sería falsa precisión.')
print(f'\n{classification_report(y_true, y_pred, target_names=CLASES_ES, zero_division=0)}')

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=CLASES_ES).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Matriz de confusión — ResNet-18 (test n={len(y_true)})\n'
             f'Acc {acc*100:.1f}% IC95%[{ic_acc_lo*100:.0f}–{ic_acc_hi*100:.0f}%]',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, 'matriz_confusion_f3.png'), dpi=150)
plt.show()


# =====================================================================
# BLOQUE 7-BIS — REEMPLAZA LA CELDA 16 (cuantización) · EL FIX CRÍTICO
# ---------------------------------------------------------------------
# QUÉ SE CORRIGE — DOS ERRORES GRAVES
#
# ERROR 1 · El accuracy INT8 de "100%" no medía nada.
#   El código original hacía:
#       loader_test_cpu = DataLoader(ImageFolder(RUTA_IMG,...), shuffle=False)
#       for i,(imgs,etiq) in enumerate(loader_test_cpu):
#           if i > 3: break        # 4 batches = 128 imágenes
#   ImageFolder sin barajar recorre las clases en orden alfabético y
#   'Bersih' (Limpia) tiene 311 imágenes: esas 128 son TODAS de una sola
#   clase y TODAS del conjunto de entrenamiento. De ahí el 100%.
#   El informe lo reportaba como "+2.6 pp" de mejora. La cuantización
#   NO puede mejorar el accuracy.
#   → Ahora se evalúa sobre el MISMO test (idx_test) que el FP32.
#
# ERROR 2 · La cuantización era, en la práctica, un no-op.
#       quantize_dynamic(modelo, qconfig_spec={nn.Linear}, ...)
#   ResNet-18 tiene UNA sola capa Linear (512→3 = 1.539 parámetros de
#   11.178.051 = 0,014% del modelo). Todo lo demás son Conv2d, que la
#   cuantización DINÁMICA no toca. Por eso el tamaño no bajó 1 byte.
#   El informe lo atribuía a "los metadatos compensan el ahorro":
#   explicación incorrecta. No se cuantizó nada.
#   → Se añade cuantización ESTÁTICA (PTQ) con calibración, que sí
#     cuantiza las Conv2d, y se conserva la dinámica como contraste
#     para documentar la lección.
# =====================================================================

import copy
import torch.ao.quantization as tq
from torchvision.models.quantization import resnet18 as qresnet18

torch.backends.quantized.engine = 'fbgemm'   # x86 (Colab)
CPU = torch.device('cpu')

def tam_mb(m, ruta):
    torch.save(m.state_dict(), ruta)
    return os.path.getsize(ruta) / 1e6

def medir_latencia(m, n=30, descarte=10):
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        for _ in range(descarte):     # calentamiento real
            m(x)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); m(x); ts.append((time.perf_counter()-t0)*1000)
    return float(np.median(ts))       # mediana: robusta a outliers

def acc_en_test(m, dispositivo=CPU):
    yt, yp, _ = evaluar(m, loader_test, dispositivo, usar_amp=False)
    return (yt == yp).mean()

# ── A · Modelo base FP32 en CPU ─────────────────────────────────────
modelo_fp32 = copy.deepcopy(modelo).to(CPU).eval()
tam_fp32 = tam_mb(modelo_fp32, os.path.join(RUTA_SALIDA, 'modelo_fp32.pth'))
lat_fp32 = medir_latencia(modelo_fp32)
acc_fp32 = acc_en_test(modelo_fp32)

# ── B · Cuantización DINÁMICA (se conserva como contraste) ──────────
modelo_din = tq.quantize_dynamic(copy.deepcopy(modelo_fp32),
                                 qconfig_spec={nn.Linear}, dtype=torch.qint8)
tam_din = tam_mb(modelo_din, os.path.join(RUTA_SALIDA, 'modelo_int8_dinamico.pth'))
lat_din = medir_latencia(modelo_din)
acc_din = acc_en_test(modelo_din)

n_linear = sum(p.numel() for m_ in modelo_fp32.modules()
               if isinstance(m_, nn.Linear) for p in m_.parameters())
n_total  = sum(p.numel() for p in modelo_fp32.parameters())
print('=== POR QUÉ LA CUANTIZACIÓN DINÁMICA NO REDUJO EL TAMAÑO ===')
print(f'Parámetros en capas Linear : {n_linear:>12,}')
print(f'Parámetros totales         : {n_total:>12,}')
print(f'Fracción cuantizable       : {n_linear/n_total*100:>12.3f}%')
print('quantize_dynamic solo alcanza Linear/LSTM/GRU. ResNet-18 es casi')
print('toda Conv2d, así que la técnica no aplica aquí. Ese es el motivo')
print('real del 0% de reducción — no los metadatos.\n')

# ── C · Cuantización ESTÁTICA (PTQ con calibración) ─────────────────
modelo_est = qresnet18(weights=None, quantize=False)
modelo_est.fc = nn.Linear(512, N_CLASES)

sd_src = modelo.state_dict()
sd_dst = {k: v for k, v in sd_src.items() if not k.startswith('fc.')}
sd_dst['fc.weight'] = sd_src['fc.1.weight']    # nuestro fc es Sequential(Dropout, Linear)
sd_dst['fc.bias']   = sd_src['fc.1.bias']
faltan = modelo_est.load_state_dict(sd_dst, strict=False)
print(f'Carga de pesos en modelo cuantizable → faltantes: {len(faltan.missing_keys)}, '
      f'inesperados: {len(faltan.unexpected_keys)}')

modelo_est.eval()
modelo_est.fuse_model(is_qat=False)            # fusiona Conv+BN+ReLU
modelo_est.qconfig = tq.get_default_qconfig('fbgemm')
tq.prepare(modelo_est, inplace=True)

print('Calibrando con imágenes de TRAIN (nunca de test)...')
loader_calib = DataLoader(Subset(dataset_ref, idx_train[:128]),
                          batch_size=32, shuffle=False)
with torch.no_grad():
    for imgs, _ in loader_calib:
        modelo_est(imgs)

tq.convert(modelo_est, inplace=True)
tam_est = tam_mb(modelo_est, os.path.join(RUTA_SALIDA, 'modelo_int8_estatico.pth'))
lat_est = medir_latencia(modelo_est)
acc_est = acc_en_test(modelo_est)

# ── D · TABLA COMPARATIVA — TODO SOBRE EL MISMO TEST ────────────────
print('\n' + '=' * 74)
print(f'TABLA DE OPTIMIZACIÓN EDGE — evaluado sobre el MISMO test (n={len(idx_test)})')
print('=' * 74)
print(f'{"Métrica":<26}{"FP32":>13}{"INT8 dinám.":>15}{"INT8 estát.":>15}')
print('-' * 74)
print(f'{"Tamaño (MB)":<26}{tam_fp32:>13.1f}{tam_din:>15.1f}{tam_est:>15.1f}')
print(f'{"Latencia mediana (ms)":<26}{lat_fp32:>13.1f}{lat_din:>15.1f}{lat_est:>15.1f}')
print(f'{"Accuracy en test (%)":<26}{acc_fp32*100:>13.1f}{acc_din*100:>15.1f}{acc_est*100:>15.1f}')
print('=' * 74)
print(f'Reducción de tamaño   FP32→INT8 estático: {(1-tam_est/tam_fp32)*100:>6.1f}%')
print(f'Reducción de latencia FP32→INT8 estático: {(1-lat_est/lat_fp32)*100:>6.1f}%')
print(f'Cambio de accuracy    FP32→INT8 estático: {(acc_est-acc_fp32)*100:>+6.1f} pp')
print('\nLas tres columnas se miden sobre el mismo conjunto de prueba, con')
print('las mismas imágenes y el mismo protocolo. Es la única forma de que')
print('la comparación signifique algo.')

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('Optimización Edge — FP32 vs INT8 dinámico vs INT8 estático',
             fontsize=12, fontweight='bold')
for a, (tit, vals) in zip(axes, [
        ('Tamaño (MB)',   [tam_fp32, tam_din, tam_est]),
        ('Latencia (ms)', [lat_fp32, lat_din, lat_est]),
        ('Accuracy (%)',  [acc_fp32*100, acc_din*100, acc_est*100])]):
    bars = a.bar(['FP32', 'INT8\ndinám.', 'INT8\nestát.'], vals,
                 color=['#C0392B', '#E8A33D', '#02C39A'], alpha=0.85)
    for b, v in zip(bars, vals):
        a.text(b.get_x()+b.get_width()/2, b.get_height()+max(vals)*0.02,
               f'{v:.1f}', ha='center', fontweight='bold')
    a.set_title(tit, fontweight='bold'); a.grid(axis='y', alpha=.3)
    a.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, 'optimizacion_edge.png'), dpi=150)
plt.show()


# =====================================================================
# BLOQUE 8-BIS — REEMPLAZA LA CELDA 18 (exportación)
# ---------------------------------------------------------------------
# QUÉ SE CORRIGE
#   (a) La bitácora afirma haber "rechazado dynamo=True y usado la ruta
#       estable". El log del notebook original demuestra lo contrario:
#         [torch.onnx] Obtain model graph ... with torch.export.export(...)
#       En PyTorch 2.11 el exportador dynamo es el PREDETERMINADO. No
#       pasar dynamo=True no equivale a usar la ruta legacy: hay que
#       pasar dynamo=False explícitamente. Se corrige y se documenta.
#   (b) El .onnx pesaba 90 KB porque los pesos iban a un sidecar
#       .onnx.data de 44,7 MB. Si solo se entrega el .onnx, no carga.
#       Se fuerza archivo único.
#   (c) El modelo .keras se guardaba SIN ENTRENAR (pesos ImageNet +
#       cabeza aleatoria). Se entrena brevemente para que el archivo
#       represente algo, y se declara que es una implementación
#       equivalente, no el mismo objeto.
# =====================================================================

import onnx, onnxruntime as ort

modelo_export = copy.deepcopy(modelo).to(CPU).eval()
X_dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

# --- 1 · .pth ---
ruta_pth = os.path.join(RUTA_SALIDA, 'Centinela_Fase3_ResNet18.pth')
torch.save({'state_dict': modelo_export.state_dict(), 'arquitectura': 'ResNet18',
            'n_clases': N_CLASES, 'clases': CLASES_ES, 'img_size': IMG_SIZE,
            'acc_test': float(acc), 'auc_test': float(auc),
            'ic95_acc': [float(ic_acc_lo), float(ic_acc_hi)]}, ruta_pth)
print(f'1. .pth   {os.path.getsize(ruta_pth)/1e6:>6.1f} MB')

# --- 2 · .onnx por la RUTA ESTABLE explícita (dynamo=False) ---
ruta_onnx = os.path.join(RUTA_SALIDA, 'Centinela_Fase3_ResNet18.onnx')
try:
    torch.onnx.export(
        modelo_export, X_dummy, ruta_onnx,
        export_params=True, opset_version=18,
        input_names=['entrada'], output_names=['salida'],
        dynamic_axes={'entrada': {0: 'lote'}, 'salida': {0: 'lote'}},
        dynamo=False)                      # ← ruta estable EXPLÍCITA
    print('   Exportado por la ruta estable (dynamo=False)')
except TypeError:
    torch.onnx.export(
        modelo_export, X_dummy, ruta_onnx,
        export_params=True, opset_version=18,
        input_names=['entrada'], output_names=['salida'],
        dynamic_axes={'entrada': {0: 'lote'}, 'salida': {0: 'lote'}})
    print('   Esta versión no acepta el argumento dynamo; se usó el default')

# Forzar archivo ÚNICO (sin sidecar .onnx.data)
_m = onnx.load(ruta_onnx)
onnx.save_model(_m, ruta_onnx, save_as_external_data=False)
onnx.checker.check_model(onnx.load(ruta_onnx))
sidecar = ruta_onnx + '.data'
if os.path.exists(sidecar):
    os.remove(sidecar)
print(f'2. .onnx  {os.path.getsize(ruta_onnx)/1e6:>6.1f} MB (archivo único, válido)')

# --- 3 · .keras ENTRENADO ---
def crear_keras(n_clases, img_size=224):
    base = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet',
                                            input_shape=(img_size, img_size, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    return tf.keras.Model(base.input, tf.keras.layers.Dense(n_clases, activation='softmax')(x))

def ds_keras(indices, barajar):
    rutas = [dataset_ref.samples[i][0] for i in indices]
    ys    = [dataset_ref.samples[i][1] for i in indices]
    d = tf.data.Dataset.from_tensor_slices((rutas, ys))
    if barajar:
        d = d.shuffle(len(rutas), seed=SEMILLA)
    return (d.map(cargar_imagen_tf, num_parallel_calls=AUTOTUNE)
             .cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))

modelo_keras = crear_keras(N_CLASES)
modelo_keras.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('\n3. Entrenando la implementación Keras (5 épocas)...')
modelo_keras.fit(ds_keras(idx_train, True), validation_data=ds_keras(idx_val, False),
                 epochs=5, verbose=2)
loss_k, acc_k = modelo_keras.evaluate(ds_keras(idx_test, False), verbose=0)
ruta_keras = os.path.join(RUTA_SALIDA, 'Centinela_Fase3_ResNet.keras')
modelo_keras.save(ruta_keras)
print(f'   .keras {os.path.getsize(ruta_keras)/1e6:>6.1f} MB | acc test Keras: {acc_k*100:.1f}%')
print('   NOTA: .pth y .keras son DOS IMPLEMENTACIONES EQUIVALENTES')
print('   (ResNet-18 PyTorch / ResNet50V2 Keras), no el mismo objeto')
print('   serializado dos veces. Sus accuracies no tienen por qué coincidir.')

# --- Verificación de consistencia PyTorch ↔ ONNX ---
with torch.no_grad():
    logits_torch = modelo_export(X_dummy).numpy()
sess = ort.InferenceSession(ruta_onnx)
logits_onnx = sess.run(None, {'entrada': X_dummy.numpy()})[0]

consistente = np.allclose(logits_torch, logits_onnx, atol=1e-4)
diff_max = float(np.abs(logits_torch - logits_onnx).max())
print(f'\n=== VERIFICACIÓN DE CONSISTENCIA (sobre LOGITS, no softmax) ===')
print(f'PyTorch vs ONNX   : {"CONSISTENTE" if consistente else "INCONSISTENTE"}')
print(f'Diferencia máxima : {diff_max:.2e}   (umbral atol=1e-4)')

# Verificación adicional sobre imágenes reales (más exigente que un dummy)
imgs_reales, _ = next(iter(loader_test))
with torch.no_grad():
    lt = modelo_export(imgs_reales).numpy()
sess_d = ort.InferenceSession(ruta_onnx)
lo = sess_d.run(None, {'entrada': imgs_reales.numpy()})[0]
print(f'Sobre {len(imgs_reales)} imágenes reales: '
      f'{"CONSISTENTE" if np.allclose(lt, lo, atol=1e-4) else "INCONSISTENTE"} '
      f'(dif. máx {np.abs(lt-lo).max():.2e})')
print(f'Acuerdo de predicciones: {(lt.argmax(1) == lo.argmax(1)).mean()*100:.1f}%')


# =====================================================================
# BLOQUE 10 — NUEVO · RESUMEN DE CIFRAS PARA EL INFORME
# ---------------------------------------------------------------------
# Todo lo que va al informe sale de aquí. Si un número no aparece en
# esta salida, NO se reporta. (Lección de la Fase 2.)
# =====================================================================

resumen = {
    'GPU':                      torch.cuda.get_device_name(0),
    'PyTorch':                  torch.__version__,
    'TensorFlow':               tf.__version__,
    'n_train / n_val / n_test': f'{len(idx_train)} / {len(idx_val)} / {len(idx_test)}',
    '--- PIPELINE ---':         '',
    'PyTorch sin optim (ms)':   round(t_sin*1000, 1),
    'PyTorch con optim (ms)':   round(t_con*1000, 1),
    'tf.data ép.2 sin optim':   round(t_tf_sin_e2*1000, 1),
    'tf.data ép.2 con optim':   round(t_tf_con_e2*1000, 1),
    '--- AUGMENTATION ---':     '',
    'acc_val sin augment (%)':  round(acc_sin_aug*100, 1),
    'acc_val con augment (%)':  round(acc_con_aug*100, 1),
    '--- GPU (MEDIDO) ---':     '',
    'VRAM FP32 (MB)':           round(vram_fp32_med),
    'VRAM FP16 (MB)':           round(vram_fp16_med),
    'Ahorro VRAM (%)':          round(ahorro_vram, 1),
    'Tiempo FP32 (s/época)':    round(t_fp32_med, 2),
    'Tiempo FP16 (s/época)':    round(t_fp16_med, 2),
    'Aceleración (x)':          round(acel, 2),
    '--- TEST ---':             '',
    'Accuracy (%)':             round(acc*100, 1),
    'IC95% accuracy':           f'[{ic_acc_lo*100:.1f} – {ic_acc_hi*100:.1f}]',
    'ROC-AUC':                  round(auc, 3),
    'IC95% ROC-AUC':            f'[{ic_auc_lo:.3f} – {ic_auc_hi:.3f}]',
    '--- EDGE ---':             '',
    'Tamaño FP32 (MB)':         round(tam_fp32, 1),
    'Tamaño INT8 dinám (MB)':   round(tam_din, 1),
    'Tamaño INT8 estát (MB)':   round(tam_est, 1),
    'Latencia FP32 (ms)':       round(lat_fp32, 1),
    'Latencia INT8 estát (ms)': round(lat_est, 1),
    'Accuracy INT8 estát (%)':  round(acc_est*100, 1),
    '% params cuantizables din': round(n_linear/n_total*100, 3),
    '--- EXPORTACIÓN ---':      '',
    'Consistencia ONNX':        bool(consistente),
    'Diferencia máx ONNX':      f'{diff_max:.2e}',
    'Accuracy Keras (%)':       round(acc_k*100, 1),
}

print('=' * 62)
print('CIFRAS VERIFICADAS PARA EL INFORME — FASE 3')
print('=' * 62)
for k, v in resumen.items():
    print(f'{k:<30}{v}' if v != '' else f'\n{k}')
print('=' * 62)

import json
with open(os.path.join(RUTA_SALIDA, 'cifras_informe.json'), 'w', encoding='utf8') as f:
    json.dump({k: v for k, v in resumen.items() if v != ''}, f,
              indent=2, ensure_ascii=False)
print('\nGuardado: cifras_informe.json  ← única fuente de verdad del informe')
