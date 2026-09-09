# Bitácora completa de sesión — 2026-09-08/09

Registro íntegro de la sesión de análisis: qué se pidió, qué se leyó, qué se
ejecutó, qué se midió, qué se concluyó, qué se corrigió y qué queda pendiente.
Se escribe con detalle deliberado para que la sesión sea reconstruible sin
acceso al historial de chat.

- **Rama:** `claude/segmentacion-pipeline-analysis-d1k1ff`
- **Commit base auditado:** `985a920` (`feat: segmenter completed`)
- **Commit del análisis:** `b28ef01` (`docs: revisión profunda del pipeline de segmentación`)
- **Entorno:** contenedor remoto efímero, sin dataset materializado, sin GPU,
  sin `torch` ni `ultralytics` instalados. Todo lo ejecutado usó sólo
  `numpy`, `Pillow`, `pyyaml`, `pandas`, `pytest` y `ruff`.

## 0. Índice

1. Encargo y método
2. Estado del repositorio al iniciar
3. Auditoría profunda: los 18 hallazgos
4. Verificaciones empíricas ejecutadas
5. La otra auditoría (`audit-segmenter-september`) y su comparación
6. Problemas encontrados en las correcciones de esa rama
7. Fondos mixtos en el dataset del clasificador
8. Diagnóstico del experimento fallido de clasificación
9. Literatura consultada
10. EfficientNet-lite0
11. Plan consolidado
12. Correcciones a afirmaciones propias
13. Decisiones pendientes
14. Fuentes

---

## 1. Encargo y método

El encargo inicial fue: *"Analiza a profundidad todo el pipeline de
segmentación, investiga en internet mejoras, buenas prácticas, errores,
inconsistencias, etc. Investiga también sobre casos de uso en tópicos
similares, en papers sobre clasificación o detección de enfermedades
foliares."*

Método aplicado:

1. lectura completa de `src/segmentation/`, `src/preprocessing/`,
   `src/evaluation/`, `src/training/segmentation_preflight.py`,
   `cloud_training/run_ultralytics.py`, `config/segmentation.yaml`, `Makefile`
   y la documentación de `docs/es/`;
2. lectura en superficie de `src/data/` (~7 900 líneas de consolidación,
   auditoría, splits y normalización JPEG);
3. ejecución de la suite, `ruff check` y `ruff format --check`;
4. **verificación empírica** de cada afirmación cuantitativa, ejecutando el
   código real del repositorio contra entradas sintéticas construidas para el
   caso (§4);
5. investigación bibliográfica dirigida (§9).

Limitaciones honestas: sin dataset, sin GPU y sin Ultralytics, los caminos que
tocan imágenes reales, entrenamiento o Modal se auditaron por lectura.

## 2. Estado del repositorio al iniciar

Ramas presentes al comenzar: `master` (`985a920`) y la rama de trabajo, a la
par. La rama `origin/audit-segmenter-september` **no existía en el remoto** en
ese momento; apareció al hacer `git fetch --prune` más adelante en la sesión.

Mapa del pipeline reconstruido:

| Etapa | Módulos | Estado |
|---|---|---|
| Consolidación de fuentes | `segmentation_consolidation.py`, `segmentation_audit.py`, `jpeg_normalization.py` | Congelado, lock `7a4a5c08` |
| Revisión humana y finalización | `segmentation_review.py`, `segmentation_review_preview.py`, `segmentation_finalization.py` | Congelado |
| Splits agrupados | `segmentation_split.py` | Semilla 42, 809/173/173 |
| Preflight | `src/training/segmentation_preflight.py` | Verifica locks + smoke loader |
| Empaquetado y entrenamiento | `scripts/package/`, `cloud_training/`, `modal_training.py` | D-01 completado en Modal |
| Inferencia | `src/segmentation/leaf_segmenter.py` | Wrapper Ultralytics 8.4.104 exacta |
| Selección de hoja | `segmented_leaf_processor.py` | `0.45·área + 0.35·centro + 0.20·conf` |
| Máscara y salida | `leaf_mask.py`, `letterbox.py` | 4 perfiles, fondo RGB exacto |
| Quality gate | `src/segmentation/quality.py` | v1.0.0, calibrado sobre 42 imágenes |
| Métricas downstream | `segmentation_downstream.py` | Recall de píxel priorizado |

Datos relevantes del corpus: 1 155 imágenes de dos fuentes, `corn` (155
imágenes a 224×224) y `corn_leaf_diseases_classification` (1 000 imágenes, hasta
3048×4064). 1 035 grupos indivisibles tras dedupe perceptual. Piloto externo de
100 imágenes retenido.

Resultados vigentes de D-01 (`mosaic=0.0`, semilla 42, A10, ~24 min):
Mask mAP50-95 `0.94404`, Mask mAP50 `0.97326`, Box mAP50-95 `0.94343`. Prueba
end-to-end sobre 150 imágenes de enfermedades de `val`: 150 `reliable`, IoU
medio `0.98122`, Dice `0.99046`, recall de píxel `0.99375`, peor tejido
recortado `11.41 %`.

## 3. Auditoría profunda: los 18 hallazgos

Detalle completo en
[Revisión profunda del pipeline](segmentation-pipeline-deep-review.md). Resumen
con el estado que tenían al momento de escribirse (commit `985a920`):

### Severidad alta

- **H-01 — La evaluación de `test` mide un pipeline distinto al de producción.**
  `run_ultralytics.py:988-1041` alimenta `evaluate_downstream()` con las
  etiquetas de `model.val(save_txt=True)`. El default de `conf` en modo `val`
  de Ultralytics es **0.001** (0.25 en `predict`), y guarda **todas** las
  instancias. Producción usa `selection_confidence_threshold: 0.50` y
  selecciona **una sola** hoja. Consecuencia: `leaf_pixel_recall` inflado por
  construcción, `over_segmentation_ratio` inflado en sentido contrario, y
  `fallback_rate` / `images_without_detection_rate` ≈0 estructuralmente.
  Existen dos métricas homónimas con semántica incompatible: la de D-01 usó el
  wrapper real, la de `test` no.

- **H-02 — El gate de `test` clavado al checkpoint baseline.**
  `run_ultralytics.py:51-53` y `465-477` exigían
  `segmenter/yolo26n_seg_baseline/weights/best.pt` con SHA `4f66456d…`,
  mientras el candidato es D-01 con SHA `a2bf4f20…`. El paso 4 del protocolo de
  mejora estaba bloqueado por código.

- **H-03 — Deriva de umbrales del quality gate en tres sitios.**
  YAML `0.25 / 0.80`, defaults del dataclass `0.50 / 0.70`, documentación
  `0.50 / 0.70`. Y `assess_segmentation(result)` sin `quality_gate=` usaba los
  defaults **pre-calibración**. Además `reject_multiple_eligible` valía `false`
  en YAML y `True` en la función.

- **H-04 — `normalized_perimeter` no es invariante a resolución sobre bordes
  reales.** Verificado empíricamente (§4.1). El dataset mezcla 224×224 con
  3048×4064, así que el umbral `8.0` es sistemáticamente más estricto con las
  imágenes grandes. Y el borde rugoso es la firma del daño por cogollero: **el
  gate está sesgado contra los casos diagnósticamente críticos**.

- **H-05 — El gate geométrico colapsa a un solo criterio.** La regla
  `suspicious_large_mask_geometry` exige tres condiciones en AND; dos se
  cumplen casi siempre en una hoja de maíz diagonal (verificado en §4.2), así
  que de facto sólo decide `normalized_perimeter`, el estadístico frágil de
  H-04.

- **H-06 — Riesgo de duplicados por inferencia end-to-end de YOLO26.**
  YOLO26 es NMS-free por defecto; `iou_threshold: 0.70` y `max_detections: 20`
  pueden ser parámetros muertos, y `to_metadata()` reporta
  `"nms": "class_aware"` como literal fijo. Hay reporte abierto de duplicados
  con ~100 % de solape en modelos `yolo26*-seg` end2end. Dos duplicados dan
  margen ≈0 → `ambiguous_instance_score_margin` → un `reliable` correcto se
  convierte en `uncertain`.

### Severidad media

- **H-07 — `crop_mask_letterbox` reintroduce halo bilineal.** Verificado en
  §4.3. Contradice `apply_leaf_mask` (RGB(0,0,0) exacto) y el uso deliberado de
  PNG en `save_debug_artifacts` para evitar ringing.

- **H-08 — `rasterize_polygons` deforma a cuadrado y usa `size - 1`.**
  La deformación es inocua para IoU en el continuo pero degrada la
  discretización en imágenes muy apaisadas; `size - 1` encoge ambas máscaras
  ~0.3 % y sesga `truth_area_fraction`, usado por `_area_bin`.

- **H-09 — El selector puede elegir la hoja equivocada en escenas multi-hoja.**
  El score no tiene ninguna señal relacionada con síntomas. La política es
  defendible pero es una suposición de encuadre no verificada.

- **H-10 — Aumentos por defecto inadecuados para hojas.** Ningún perfil fija
  `degrees`, `flipud`, `mask_ratio`, `overlap_mask`, `scale`, `hsv_*` ni
  `close_mosaic`. Con `mask_ratio=4` el GT se rasteriza a 160×160 para la
  pérdida siendo la fidelidad de contorno la métrica prioritaria; y sin
  rotación ni volteo vertical el modelo hereda el prior de orientación del
  corpus de laboratorio.

- **H-11 — Calibración del gate evaluada sobre la misma muestra en que se
  ajusta.** 640 candidatos (2×4×4×4×5) sobre 42 filas con 24 positivos.
  "24/24 GOOD, 0 falsos confiables" es métrica de entrenamiento; el intervalo
  de Wilson al 95 % para 0/18 llega a ~18 %.

- **H-12 — La identidad del checkpoint de inferencia no está fijada.**
  `UltralyticsLeafSegmenter` calcula `checkpoint_sha256` pero nunca lo compara
  contra un valor esperado, en un repo donde todo lo demás tiene su
  `EXPECTED_*`.

### Severidad baja

- **H-13** — `to_metadata()` devuelve `"segmenter_model": "yolo26n-seg"` como
  literal; miente si se promueve D-04 (`yolo26s-seg`).
- **H-14** — `import torch` a nivel de módulo en `segmentation_preflight.py`,
  importado por el helper de targets de sólo lectura.
- **H-15** — `make check` corre sólo `pyright`; `ruff format` no se verifica
  en ninguna parte (47 archivos pendientes en ese momento).
- **H-16** — Dos tests no herméticos y tres módulos no colectables sin `torch`.
- **H-17** — Contrato 183 vs 182 instancias abierto, bloqueando la evaluación.
- **H-18** — `_center_proximity` usa el centroide de la máscara, inestable para
  hojas en "L" o muy curvadas; el centro del bbox sería más estable.

### Lo que estaba bien resuelto

Registrado porque son fallos habituales ya cubiertos: splits agrupados con
dedupe perceptual y variantes Roboflow; piloto externo retenido; métricas
orientadas a recall de píxel en vez de mAP con justificación correcta en el
docstring; desagregación por fuente, resolución, orientación, contacto con
borde y número de hojas; guards de confirmación literal separados; escritura de
artefactos con `open(..., "x")`; y `binary_mask_array` rechazando valores
intermedios para detectar máscaras redimensionadas por interpolación.

## 4. Verificaciones empíricas ejecutadas

Todo lo de esta sección se ejecutó contra el código real del repositorio.

### 4.1 `normalized_perimeter` y resolución (soporta H-04)

Se rasterizó la misma forma a cuatro resoluciones, con y sin rugosidad de borde
del 2 %:

| Resolución | Borde liso | Borde rugoso (2 %) |
|---:|---:|---:|
| 224 | 5.488 | 7.458 |
| 640 | 5.520 | 8.155 |
| 1280 | 5.514 | 8.245 |
| 3048 | 5.509 | 8.291 |

Conclusión: invariante para bordes lisos, **creciente con la resolución para
bordes reales**. La misma forma cruza el umbral `8.0` sólo por resolución.

Script de reproducción:

```python
import math, numpy as np
from PIL import Image, ImageDraw
from src.preprocessing.leaf_mask import mask_geometry

def leaf(n, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    img = Image.new("L", (n, n), 0); d = ImageDraw.Draw(img); pts = []
    for t in np.linspace(0, 2*math.pi, 400, endpoint=False):
        r = 0.42*(1+0.35*math.cos(3*t))
        if jitter: r *= (1 + jitter*rng.normal())
        pts.append(((0.5+r*math.cos(t))*n, (0.5+0.55*r*math.sin(t))*n))
    d.polygon(pts, fill=255); return img

for n in (224, 640, 1280, 3048):
    print(n, round(mask_geometry(leaf(n)).normalized_perimeter, 3),
             round(mask_geometry(leaf(n, jitter=0.02, seed=1)).normalized_perimeter, 3))
```

### 4.2 Hoja diagonal y `mask_bbox_ratio` (soporta H-05)

```
diagonal leaf: area_ratio=0.123  mask_bbox_ratio=0.166  normalized_perimeter=9.81
```

`mask_bbox_ratio` de una hoja diagonal legítima queda muy por debajo del umbral
`0.80`, así que esa condición no discrimina.

### 4.3 Halo del letterbox (soporta H-07)

Recorte ya enmascarado a negro exacto, pasado por
`letterbox_image(..., resample=BILINEAR)` a 224×224:

```
173 colores únicos; 729 píxeles que no son ni negro puro ni color de hoja
```

### 4.4 Estado de verificación del repo en `985a920`

```
python -m pytest        297 passed, 2 failed  (fallos por entorno)
                        3 módulos no colectables sin torch
python -m ruff check .  All checks passed
python -m ruff format --check .  47 archivos se reformatearían
```

Los dos fallos: `test_status_is_read_only` requiere
`data/leaf_detection/.../dataset_lock.json` (gitignorado);
`test_package_verify_rejects_bad_sha_before_extraction` requiere `torch`
(consecuencia de H-14).

### 4.5 Idempotencia del enmascarado y el atajo "no-negro" (§7)

```
idempotente (imagen ya negra -> mask_black): True
IoU del atajo "hoja = píxel no negro" en imagen ya enmascarada: 1.000000
IoU del mismo atajo sobre fondo de suelo:                        0.0879
```

### 4.6 Detector de tipo de fondo (§7)

Heurística basada sólo en el anillo de borde de la imagen:

```python
def background_profile(img, *, ring=0.04, near_black=12):
    a = np.asarray(img.convert("RGB"), np.int16)
    h, w, _ = a.shape
    t = max(1, int(round(min(h, w) * ring)))
    r = np.concatenate([a[:t].reshape(-1,3), a[-t:].reshape(-1,3),
                        a[:, :t].reshape(-1,3), a[:, -t:].reshape(-1,3)])
    black_frac = float((r.max(1) <= near_black).mean())
    chroma  = float(np.mean(r.max(1) - r.min(1)))
    texture = float(r.std(0).mean())
    if black_frac >= 0.90:                kind = "premasked_black"
    elif chroma <= 25 and texture <= 30:  kind = "uniform_plain"
    else:                                 kind = "field_complex"
    return {"kind": kind, "border_black_frac": black_frac,
            "image_black_frac": float((a.max(2) <= near_black).mean()),
            "border_chroma": chroma, "border_texture": texture}
```

Resultados sobre cuatro casos sintéticos:

```
ya enmascarada (negro)   -> premasked_black  border_black_frac=1.00
fondo liso de estudio    -> uniform_plain    chroma=0.0   texture=0.0
suelo gris texturado     -> field_complex    chroma=30.5  texture=18.0
campo (vegetación)       -> field_complex    chroma=80.1  texture=33.8
```

Los umbrales `25` y `30` son provisionales y hay que calibrarlos sobre el
histograma real. `border_black_frac` es prácticamente binario y no lo necesita.

### 4.7 Ceguera del quality gate ante sub-segmentación (§8)

Ejecutado con el `SegmentedLeafProcessor` real y los umbrales calibrados
(`large_mask_area_ratio=0.25`, `min_large_mask_bbox_ratio=0.80`):

| Caso | status | area | recall de hoja |
|---|---|---:|---:|
| máscara correcta | `reliable` | 0.0879 | 1.000 |
| manchón (~1.5 % del área) | `reliable` | 0.0150 | 0.145 |
| manchón mínimo (~1.2 %) | `reliable` | 0.0126 | 0.112 |

**El gate declara `reliable` una máscara que conserva el 11 % de la hoja.**
Causas: `min_mask_area_ratio: 0.01`; ausencia total de comprobación de
sub-segmentación; y el colapso a `normalized_perimeter` de H-05, agravado
porque `connected_components` vale siempre 1 por construcción.

### 4.8 Normalización del color de relleno (§10)

Con media/desviación de ImageNet:

```
negro (0,0,0)                 -> [-2.12, -2.04, -1.80]
gris medio (128)              -> [ 0.07,  0.21,  0.43]
media ImageNet (124,116,104)  -> [ 0.006, -0.005, 0.008]
blanco (255)                  -> [ 2.25,  2.43,  2.64]
```

El negro puro **no es neutro**: entra como constante fuerte cerca del extremo
del rango en toda el área enmascarada.

## 5. La otra auditoría: `audit-segmenter-september`

Descubierta al hacer `git fetch --prune`. Dos commits sobre `master`, 40
archivos, +2 526/−660 líneas:

- `55a08ab` — `docs: auditoría del segmentador y comparación con el clasificador`
  (752 líneas, `docs/es/leaf-detection/auditoria-segmentador-2026-09.md`)
- `875013d` — `fix: corregir validacion en Modal, snapshot de paquetes y promover checkpoint`

Fecha de esa auditoría: **2026-09-06**, mismo commit `985a920`, dos días antes
de la mía. Su alcance declarado incluye además la comparación contra
`maize-doctor-classifier` en `ae5d1ed`.

### 5.1 Sus hallazgos

| # | Severidad | Hallazgo |
|---|---|---|
| B-01 | Bloqueante | Conteo de instancias de test congelado en 183 vs 182 efectivas |
| B-02 | Bloqueante | Sin promoción del checkpoint entrenado al que consume la inferencia |
| A-01 | Alta | Umbrales del quality gate en código ≠ los calibrados en YAML |
| A-02 | Alta | `reject_multiple_eligible` con defaults contradictorios (tres verdades) |
| A-03 | Alta | Pesos iniciales sin SHA-256 esperado (único eslabón sin pin) |
| A-04 | Alta | Los experimentos no pueden evaluarse sobre `test` |
| A-05 | Alta | Artefactos de runtime de Modal escritos fuera del árbol descargable |
| A-06 | Alta | La auditoría de fiabilidad exige el corpus de 19 GB del clasificador |
| M-01 | Media | Normalización EXIF inconsistente entre calibración y auditoría |
| M-02 | Media | `fallback` y `not detected` son el mismo booleano |
| M-03 | Media | `_distance` compara escalas incompatibles en la calibración |
| M-04 | Media | `FORCE_INTERNAL_TEST_RERUN=1` es una salida muerta |
| M-05 | Media | `modal_training.py` fuera de `make lint` y `make check` |
| M-06 | Media | `python -m pytest` no pasa en un checkout limpio |
| M-07 | Media | `target_size` de handoff hardcodeado en `(224, 224)` |
| M-08 | Media | `torch` importado en el camino "read-only" del preflight |
| B-03 | Baja | `connected_components` no puede informar nada por construcción |
| B-04 | Baja | Comprobación `np.isfinite` muerta sobre un array `uint8` |
| B-05 | Baja | `max(0.0, ...)` muerto en `over_segmentation_ratio` |
| B-06 | Baja | `LetterboxResult` mezcla `(alto, ancho)` y `(ancho, alto)` |
| B-07 | Baja | `make fmt` reformatearía 26 archivos |
| B-08 | Baja | `validate` y `test` son el mismo comando con distinta descripción |
| B-09 | Baja | `model` del config final congelado no describe la inicialización real |

Más: duplicación de `sha256()` en cinco módulos y de `project_path()` en tres.

### 5.2 Estado de mis 18 hallazgos tras su rama

| # | Estado | Su equivalente |
|---|---|---|
| H-02 | ✅ Resuelto | B-02 + A-04 |
| H-03 | ✅ Resuelto (con regresión, §6.1) | A-01 + A-02 |
| H-14 | ✅ Resuelto | M-08 |
| H-17 | ✅ Resuelto | B-01 |
| H-12 | ⚠️ Parcial: existe `promoted_checkpoint.json`, pero `UltralyticsLeafSegmenter` sigue sin verificar nada al cargar y el YAML no declara `checkpoint_sha256` | — |
| H-15 | ⚠️ Parcial: `LINT_PATHS` ya cubre `modal_training.py` y `tests/`, pero `check:` sigue siendo sólo pyright | M-05, B-07 |
| H-16 | ⚠️ Parcial: ver §6.3 | M-06 |
| H-01 | 🔴 Abierto | — |
| H-04, H-05, H-06, H-07, H-08, H-09, H-10, H-13, H-18 | 🔴 Abiertos | — |
| H-11 | 🔴 Abierto (mejoraron `_distance`, no el sobreajuste) | M-03 parcial |

**H-01 sigue siendo el más grave y nadie lo tocó.** Verificado: `evaluate_mode`
conserva `model.val(..., save_txt=True)` alimentando `evaluate_downstream`, y
`validate_yolo26n_seg.yaml` sigue sin fijar `conf`. Su M-02 eliminó
`fallback_rate` —correcto, era complementario de `detected`— pero eso quita un
síntoma sin tocar la causa.

### 5.3 Lo que ellos vieron y yo no

- **A-03** es el mejor hallazgo de las dos auditorías: `yolo26n-seg.pt` se
  descargaba de internet y su SHA se *registraba* en vez de *verificarse*.
  Ya corregido con `EXPECTED_INITIAL_WEIGHTS_SHA256`.
- **A-06**: `make leaf-segmentation-reliability-audit`, comando de primera
  línea del README, exige `DATASET_ROOT/clean/` del clasificador.
- **M-01**: la calibración del umbral no aplica `exif_transpose` y la auditoría
  sí. **Debilita directamente H-11**: el umbral calibrado no describe el sistema
  auditado.
- **B-03** corrige mi H-05: propuse `connected_components` y
  `largest_component_ratio` como señales ortogonales para robustecer el gate, y
  son **constantes** (1 y 1.0) porque la máscara viene de
  `rasterize_instance_polygon`, que rellena un único polígono simple. La
  recomendación sigue en pie sólo con `border_contact_count` y
  `bbox_aspect_ratio`; para recuperar fragmentación habría que medirla antes de
  rasterizar, sobre los contornos de `masks.xy`.
- Además: la §4 de su documento compara con el clasificador y encuentra que el
  segmentador **no tiene exportación alguna** (ni ONNX, ni TFLite, ni promoción
  a la app), lo que lo hace no desplegable hoy.

### 5.4 Correcciones aplicadas en `875013d`

- `src/training/checkpoint_promotion.py` + `scripts/pipeline/promote_leaf_segmentation_checkpoint.py`
  (promoción registrada con re-hash y manifiesto).
- `src/evaluation/split_instance_counts.py` (separa anotaciones de instancias
  efectivas del cargador).
- Gate de evaluación parametrizado por identidad de run, ya no por constante.
- `EXPECTED_INITIAL_WEIGHTS_SHA256` verificado en `verified_weights()`.
- `FORCE_INTERNAL_TEST_RERUN=1` ahora funcional: archiva los artefactos previos
  en `segmenter_evaluation/superseded/` con marca de tiempo.
- Defaults del quality gate alineados con el YAML (`0.25` / `0.80`).
- `target_size` declarado en `config/segmentation.yaml`.
- `_distance` normalizado por rango de rejilla; `assert` sustituido por
  `ValueError`.
- `LetterboxResult` renombrado a `*_width_height`; `np.isfinite` y `max(0.0,…)`
  muertos retirados; `fallback` eliminado de las métricas downstream.
- `import torch` diferido; `tests/conftest.py` con marcadores
  `requires_dataset` / `requires_pilot`; `LINT_PATHS` ampliado.
- `validate.sh` delega en `evaluate_test.sh`.
- Snapshot de distribuciones ignora paquetes vendorizados de setuptools.

## 6. Problemas encontrados en las correcciones de esa rama

Verificados ejecutando su rama en un worktree aparte.

### 6.1 Regresión en la auditoría de fiabilidad

Al unificar `reject_multiple_eligible` cambiaron el default de
`assess_segmentation_legacy` de `True` a `False`. Esa función existe para
*reproducir la política previa al quality gate* y su docstring lo sigue
diciendo. `audit_segmentation_reliability.py:315` la llama **sin kwargs**, así
que el "gate legacy" del before/after ya no rechaza multi-elegibles: pierde el
contraste que la comparación mide. Y el test nuevo
`test_reject_multiple_eligible_has_a_single_default` **fija ese default por
contrato**, protegiendo la regresión. La auditoría congelada existente se
produjo con `True`.

Arreglo propuesto: revertir el default de la función *legacy* a `True` y
excluirla de ese test, o pasar el valor explícito desde el script de auditoría.

### 6.2 La propiedad de "test de un solo uso" quedó más blanda

`SEGMENTATION_EXPECTED_BEST_SHA256` puede sobrescribir el SHA esperado y
`FORCE_INTERNAL_TEST_RERUN=1` archiva y re-ejecuta. Ambas son explícitas y
dejan rastro, y desbloquean A-04/M-04, pero el neto es que dos variables de
entorno bastan para repetir el test con otro checkpoint. Decisión legítima que
debería quedar escrita como tal.

### 6.3 M-06 no está cerrado

En un checkout sin `torch`, su rama:

```
1 error de colección  (tests/package/test_run_ultralytics_summary.py:15 import torch)
ignorándolo:  3 failed, 332 passed, 3 skipped, 3 subtests
```

`test_status_is_read_only` sigue sin marcador `requires_dataset` y falla con
`FileNotFoundError`. Arreglaron dos de los tres módulos que rompen colección,
no el tercero.

### 6.4 B-07 empeoró en números

`ruff format --check` sobre `LINT_PATHS` pasó de 47/117 a **50/101** archivos,
porque ampliaron los paths sin correr `ruff format`. Añadieron el target
`fmt-check` pero no lo cablearon a `check:`.

### 6.5 Cambio de API pública sin deprecación

`LetterboxResult.original_size` → `original_size_width_height` (y
`resized_size`). Nada en este repo los consume, pero `LetterboxResult` está en
el `__all__` de `src/preprocessing`, así que si el proyecto del clasificador los
usa, rompe en silencio.

### 6.6 Recomendación de reconciliación

1. Mergear `audit-segmenter-september` a `master`, con §6.1 corregido antes.
2. Rebasar la rama de análisis sobre eso y reescribir su documento como la
   mitad de *método y modelo* de una auditoría de dos partes, con referencias
   cruzadas.

## 7. Fondos mixtos en el dataset del clasificador

Planteamiento del usuario: en el dataset del **clasificador** hay fotos que ya
vienen con fondo negro y otras con fondo gris de suelo.

### 7.1 El punto clave

**Un fondo negro ya aplicado no es un fondo, es una máscara previa.** Eso rompe
la premisa del segmentador y crea un atajo trivial: "hoja = píxel no negro" da
IoU **1.000000** en la imagen ya enmascarada y **0.0879** en la misma hoja sobre
suelo (§4.5). Si esas imágenes estuvieran en el corpus de entrenamiento del
segmentador, cada una aportaría un ~1.0 gratis que inflaría el agregado.

`apply_leaf_mask` es **idempotente** píxel a píxel sobre una imagen ya negra.
Buena noticia operativa, con un filo: la salida `mask_black` de una imagen ya
enmascarada es idéntica a la entrada, así que aguas abajo **no se puede
distinguir** "la segmentamos nosotros" de "ya venía así" de "el segmentador
falló y el fallback devolvió la original".

### 7.2 Riesgo dominante

Si el tipo de fondo correlaciona con la clase, el clasificador aprende el fondo.
Es el fallo de PlantVillage en su forma más aguda. Medir primero la tabla
cruzada `tipo_de_fondo × clase` y `tipo_de_fondo × split`.

### 7.3 Recomendaciones transversales

- Un **único fondo canónico** para el 100 % de las imágenes: así el fondo
  transporta cero información sobre la clase. Homogeneizar aguas arriba.
- **Entregar la máscara como canal aparte** (perfil `mask_alpha`: RGB original
  + máscara en el canal A) en vez de quemar el fondo. El clasificador decide
  early vs late masking, no se destruye señal, y desaparece la ambigüedad
  negro-nativo vs negro-nuestro.
- `background_kind` como columna del manifiesto, eje de estratificación de
  splits y eje de desagregación de métricas.

## 8. Diagnóstico del experimento fallido de clasificación

Datos aportados por el usuario:

- el dataset de fondos mixtos es el del **clasificador**, no el del segmentador;
- **el segmentador nunca vio fotos así** durante su entrenamiento;
- una clase con fondo gris quedó en **F1 0.20**; el segmentador sólo agregó
  "manchones negros";
- **para las demás clases las métricas quedaron peores que en la corrida sin
  segmentar**;
- roya común tiene mayoría de fotos con fondo negro (PlantVillage) y sus
  métricas no son malas en extremo;
- clasificador: **EfficientNet-lite0**.

### 8.1 Lectura

El resultado **no es una anomalía**: es lo que predice la literatura cuando la
calidad de la máscara es la restricción activa. Pero hay algo más específico:

- **Clase de fondo gris → F1 0.20.** El segmentador nunca vio suelo. Es OOD
  puro, sub-segmenta catastróficamente y deja manchones. El clasificador no
  recibe una hoja sin fondo: recibe **fragmentos de hoja sobre negro**. No se
  eliminó sesgo, se destruyó señal.
- **Roya común → no tan mala.** Para las de PlantVillage el segmentador es
  prácticamente un **no-op** (idempotencia verificada). Esas imágenes pasan casi
  intactas. **No es evidencia de que el segmentador funcione en esa clase; es
  evidencia de que no hizo nada.**
- **El daño real: corrupción correlacionada con la clase.** Unas clases pasan
  intactas y otras quedan destrozadas. Antes había un fondo espurio
  correlacionado con la clase; ahora hay eso *más* pérdida de señal en las
  clases afectadas, *más* una ambigüedad nueva: una hoja de otra clase reducida
  a manchones sobre negro se parece visualmente a una roya de PlantVillage
  nativa. Explica por qué empeoraron **todas** las clases, no sólo la gris.

### 8.2 La evidencia está en el propio código

El gate declara `reliable` una máscara que conserva el 11 % de la hoja (§4.7).
Con el gate arreglado, esas imágenes habrían caído a `fallback: original` y el
clasificador habría recibido la foto sin tocar; la pérdida se habría acotado
sola.

## 9. Literatura consultada

### 9.1 El fondo lleva señal real, no sólo sesgo

[Xiao et al., ICLR 2021 — *Noise or Signal: The Role of Image Backgrounds in
Object Recognition* (ImageNet-9)](https://arxiv.org/abs/2006.09994): los
modelos alcanzan exactitud no trivial **usando sólo el fondo**; clasifican mal
hasta el **87.5 %** de las veces con fondos adversarios aunque el primer plano
esté correcto; y los modelos más exactos dependen menos del fondo. Una ResNet-50
preentrenada de PyTorch baja al **22 %** frente a fondos adversarios en IN-9
(referencia: predecir siempre "perro" da 11 %). Lectura aplicable: quitar el
fondo también quita información que el modelo usaba legítimamente.

### 9.2 Estrategias de enmascarado

[Aniraj et al., ICCVW (OODCV) 2023 — *Masking Strategies for Background Bias
Removal in Computer Vision Models*](https://arxiv.org/abs/2308.12127): *early
masking* elimina el fondo a nivel de imagen de entrada; *late masking* enmascara
las características espaciales de alto nivel correspondientes al fondo. Ambas
mejoran el rendimiento OOD frente al baseline, y **el early masking es
consistentemente el mejor en OOD**; la mejor combinación es un ViT con
clasificación por token de parche GAP-pooled más early masking. **Trabajan con
máscaras de buena calidad**: el resultado no se transfiere a un segmentador que
falla.

### 9.3 Propagación de error en cascada

Formulación más exacta del caso: *"cuando la máscara predicha se desvía de la
verdad, la red de clasificación recibe información incompleta de la lesión
(sub-segmentación) o exceso de fondo irrelevante (sobre-segmentación), y ambas
degradan su capacidad diagnóstica"*. También documentado en imagen de tejidos
altamente multiplexada: los errores de segmentación se propagan y llevan a
interpretaciones erróneas
([PLOS Comput Biol](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013350)).

### 9.4 Sesgo de publicación en el nicho foliar

Se buscaron explícitamente resultados negativos y prácticamente no existen. Los
trabajos reportan mejoras: 92.6 % vs 80.21 % con sustracción de fondo en
semillas; DeepLabV3+ con 96.97/92.89 % de exactitud de entrenamiento/validación;
segmentación de hoja 93.27 % y severidad 92.85 %
([Frontiers 2022](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1031748/full),
[ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S277237552400131X),
[Sci Rep 2026](https://www.nature.com/articles/s41598-026-51539-2)).
**Casi ninguno reporta la calidad de la máscara ni evalúa el caso en que el
segmentador falla.** El resultado negativo del usuario no contradice la
literatura: ocupa un hueco que la literatura no cubre.

### 9.5 La excepción reciente

[*Robust Plant Disease Diagnosis with Few Target-Domain
Samples*](https://arxiv.org/pdf/2510.12909): *"la eliminación del fondo tiene un
impacto sólo limitado en el rendimiento diagnóstico con datasets grandes y de
alta resolución. Las características específicas de dominio suelen estar
embebidas en el fondo **y también dentro de la región de la hoja**"*. Enmascarar
no elimina el sesgo de dominio, sólo su mitad visible.

### 9.6 Qué recomienda la literatura en vez del enmascarado estático

[BackMix (TPAMI 2025)](https://arxiv.org/abs/2503.17717) y
[BackMix para ecocardiografía](https://arxiv.org/html/2406.19148): en lugar de
poner el fondo a negro, **cambiar el fondo por otro aleatorio**. Descorrelaciona
fondo y clase sin destruir el primer plano y generaliza mejor que el enmascarado
estático. Mismo espíritu que
[LeafGAN/LFLSeg](https://arxiv.org/abs/2002.10100), cuyo talón de Aquiles
documentado es que el segmentador no traza bien el borde con hojas solapadas o
parciales. Relacionados:
[MaskTune](https://arxiv.org/pdf/2210.00055),
[Automated Background Swapping](https://arxiv.org/pdf/2606.32018),
[Enhanced-RICAP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12504387/).

### 9.7 Brecha laboratorio→campo

Mohanty et al. (2016): caída a ~31 % al evaluar en campo modelos entrenados en
PlantVillage. Para maíz, 94.1 % en el test de PlantVillage para gray leaf spot
frente a **55.1 %** en imágenes de campo
([PMC9330607](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330607/)). Trabajo
reciente lo describe como fallo simultáneo de exactitud, calibración, predicción
selectiva y rechazo de clases desconocidas, con **eficacia limitada de las
mitigaciones estándar**
([PMC13236948](https://pmc.ncbi.nlm.nih.gov/articles/PMC13236948/)). El sesgo de
fondo de PlantVillage está documentado
([arXiv:2206.04374](https://arxiv.org/abs/2206.04374)).

### 9.8 Métricas de calidad de contorno

[Boundary IoU (Cheng et al., CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf):
más sensible que Mask IoU a errores de contorno en objetos grandes sin
sobrepenalizar los pequeños; simétrica y balanceada entre escalas. Candidata a
sustituir `normalized_perimeter` en evaluación (no en el gate en línea, porque
requiere ground truth).

### 9.9 Anotación asistida

[ARAMSAM](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1748468/full)
orquesta SAM 1 y SAM 2 para pre-etiquetado agrícola;
[SAM→YOLOv8 para hojas](https://doi.org/10.3390/agronomy15051081). SAM 2 no
supera consistentemente a SAM 1 y depende del ajuste de sus generadores
automáticos. Compatible con la regla del protocolo (prohibido usar máscaras
generadas como verdad de referencia) si se usa como pre-etiquetado que un humano
corrige y aprueba; conviene añadir `prelabel_source` al manifiesto de ingesta.

### 9.10 Estado de YOLO26

YOLO26-seg reporta hasta +3.7 mask AP sobre YOLO11 en COCO, con módulo proto
multiescala y pérdida de segmentación semántica auxiliar. Es NMS-free por
defecto. Puntos de atención:
[duplicados end2end en `yolo26*-seg`](https://github.com/ultralytics/ultralytics/issues/23685)
y los problemas históricos de
[`retina_masks` con padding no divisible por 2](https://github.com/ultralytics/ultralytics/issues/6110)
— este último ya mitigado aquí porque el wrapper valida `orig_shape` antes de
rasterizar.

### 9.11 TTA

`model.predict(augment=True)` mejora recall en varios contextos, pero degrada
cuando el objeto es pequeño respecto a la imagen. Aquí la hoja suele ocupar
>25 % del cuadro, así que el riesgo es bajo. Vale como ablación de inferencia
sin reentrenar, medida sobre `val` y sobre el subgrupo `small_leaf`. Coste ~3×.

## 10. EfficientNet-lite0

Tres características, todas consecuencia de estar optimizado para cuantización
int8, juegan en contra en este escenario
([TensorFlow Blog](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html),
[tensorflow/tpu](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)):

- **Sin bloques Squeeze-and-Excitation** (eliminados por soporte en
  aceleradores móviles). El SE es justamente el mecanismo de recalibración
  global de canales que ayudaría a detectar que una imagen es rara globalmente.
  Sin él, `lite0` depende más de estadísticas locales de textura y color.
- **Swish → ReLU6.** Con regiones negras enormes hay muchas activaciones
  muertas y estadísticas de BatchNorm desplazadas respecto al preentrenamiento.
- **~4.7 M parámetros** y stem/head fijos: poca capacidad de sobra para
  absorber un cambio de distribución en la entrada además de aprender la tarea.

Dato accionable (§4.8): con normalización de ImageNet, el negro puro entra como
`[-2.12, -2.04, -1.80]`, una constante fuerte cerca del extremo del rango.
**Rellenar con el color medio (124, 116, 104) deja esa región en ≈0**, es decir
verdaderamente neutra. Es un cambio de una línea
(`background_value` en `config/segmentation.yaml`) y merece una ablación.
Contrapartida: con gris medio se pierde la distinción entre relleno y fondo de
suelo, razón adicional para preferir la máscara como canal aparte.

## 11. Plan consolidado

### 11.1 Del segmentador (revisión profunda)

| # | Acción | Hallazgos | Esfuerzo |
|---|---|---|---|
| 1 | Resolver el contrato 183 vs 182 | H-17 | ✅ hecho en la otra rama |
| 2 | Despinnear el checkpoint de `evaluate_mode` | H-02 | ✅ hecho en la otra rama |
| 3 | Alinear defaults del gate y centralizar el loader de config | H-03 | ✅ parcial, con §6.1 pendiente |
| 4 | Fijar `checkpoint_sha256` en la ruta de inferencia | H-12 | ⚠️ falta cablear al wrapper |
| 5 | Separar `val_metrics` de `pipeline_metrics` | H-01 | 🔴 pendiente, prioritario |
| 6 | Dedupe de instancias por mask-IoU + registrar `end2end` real | H-06 | 🔴 pendiente |
| 7 | Normalizar `normalized_perimeter` y añadir señales ortogonales | H-04, H-05 | 🔴 pendiente |
| 8 | Corregir `crop_mask_letterbox` (NEAREST + remask) | H-07 | 🔴 pendiente |
| 9 | Ablaciones `D-07` (`mask_ratio=1`) y `D-08` (rotación/flip vertical) | H-10 | 🔴 pendiente |
| 10 | Repetir D-01 con semillas 7 y 1337, reportar media ± σ | — | 🔴 pendiente |
| 11 | Ronda de 300 imágenes difíciles, también como validación del gate | H-11 | 🔴 pendiente |
| 12 | Higiene: `check` con lint+format, tests herméticos | H-15, H-16 | ⚠️ parcial |

### 11.2 Del handoff al clasificador (orden recomendado)

1. **No usar la segmentación como preprocesamiento del clasificador todavía.**
   El baseline sin segmentar es el mejor resultado honesto disponible;
   congelarlo como referencia.
2. **Arreglar el gate antes de reintentar.** Subir `min_mask_area_ratio` a un
   valor derivado de la distribución real de área foliar (medir el percentil 1
   en `val`; probablemente 0.08–0.15, no 0.01) y añadir una regla explícita de
   sub-segmentación. Con `fallback: original`, lo que no supere el gate llega
   sin tocar al clasificador.
3. **Nunca enmascarar una imagen que ya venía enmascarada, ni una que el gate no
   marcó `reliable`.** Registrar `background_kind` y `segmentation_status` por
   imagen y **desagregar las métricas del clasificador por esas dos variables**.
4. **Entregar la máscara como cuarto canal** en vez de quemar el fondo.
5. **Adoptar BackMix como aumento**, no el enmascarado como preproceso: las de
   PlantVillage con fondo negro regalan la máscara, así que se pueden recomponer
   sobre fondos de suelo y campo variados.
6. **El segmentador necesita ver suelo**: es la ronda de 300 imágenes difíciles,
   y ahora hay prueba de que el subgrupo "fondo de suelo" es el que decide.

## 12. Correcciones a afirmaciones propias

Registradas explícitamente para que no se propaguen:

1. **Atribución errónea sobre early masking.** En la primera versión de la
   revisión profunda se escribió que los trabajos sobre estrategias de
   enmascarado *"reportan que en varias arquitecturas el enmascarado temprano
   degrada resultados"*, citando a Aniraj et al. Es falso: **Aniraj et al.
   concluyen que el early masking da la mejor robustez OOD**. La degradación
   citada venía de [otro paper, sobre imágenes de moda](https://arxiv.org/html/2308.09764v2),
   donde caen los backbones Swin-T y PVT de Mask R-CNN. Corregido en
   `segmentation-pipeline-deep-review.md` en el mismo commit que esta bitácora.
2. **Señales ortogonales para el gate (H-05).** Se propusieron
   `connected_components` y `largest_component_ratio`, que son constantes por
   construcción (B-03 de la otra auditoría). Ver §5.3.

## 13. Decisiones pendientes

Preguntas abiertas al cierre de la sesión, ninguna resuelta:

1. ¿Se mergea primero `audit-segmenter-september` a `master` (con §6.1
   corregido) y luego se rebasa esta rama, o al revés?
2. ¿Se implementa ya el arreglo del gate (umbral de área derivado de datos +
   regla de sub-segmentación + tests), o primero la instrumentación
   `background_kind` × `segmentation_status` para medir el tamaño exacto del
   problema en las corridas actuales?
3. ¿Se añade el diagnóstico de fondos como
   `scripts/checks/profile_background_types.py` + `src/data/background_profile.py`
   con tests? El dataset con fondos mixtos es el del clasificador, así que hay
   que decidir si eso respeta el alcance declarado en `CLAUDE.md`.
4. ¿Se abre el perfil `mask_alpha` como quinto perfil de salida?

## 14. Fuentes

- [Noise or Signal: The Role of Image Backgrounds in Object Recognition (ICLR 2021)](https://arxiv.org/abs/2006.09994) · [backgrounds_challenge](https://github.com/MadryLab/backgrounds_challenge)
- [Masking Strategies for Background Bias Removal in Computer Vision Models (ICCVW 2023)](https://arxiv.org/abs/2308.12127) · [código](https://github.com/ananthu-aniraj/masking_strategies_bias_removal)
- [The Impact of Background Removal on Performance of Neural Networks for Fashion Image Classification and Segmentation](https://arxiv.org/html/2308.09764v2)
- [Boundary IoU: Improving Object-Centric Image Segmentation Evaluation (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf)
- [Uncovering bias in the PlantVillage dataset](https://arxiv.org/abs/2206.04374)
- [Quantifying the reliability gap in cross-domain plant disease classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC13236948/)
- [Deep Learning Diagnostics of Gray Leaf Spot in Maize under Mixed Disease Field Conditions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330607/)
- [Robust Plant Disease Diagnosis with Few Target-Domain Samples](https://arxiv.org/pdf/2510.12909)
- [BackMix: Regularizing Open Set Recognition by Removing Underlying Fore-Background Priors (TPAMI 2025)](https://arxiv.org/abs/2503.17717)
- [BackMix: Mitigating Shortcut Learning in Echocardiography with Minimal Supervision](https://arxiv.org/html/2406.19148)
- [MaskTune: Mitigating Spurious Correlations by Forcing to Explore](https://arxiv.org/pdf/2210.00055)
- [Automated Background Swapping for Robustness against Spurious Backgrounds](https://arxiv.org/pdf/2606.32018)
- [Enhanced-RICAP](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12504387/)
- [LeafGAN: An Effective Data Augmentation Method for Practical Plant Disease Diagnosis](https://arxiv.org/abs/2002.10100)
- [Deep learning-based segmentation and classification of leaf images for tomato plant disease](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.1031748/full)
- [Semantic segmentation for plant leaf disease classification and damage detection](https://www.sciencedirect.com/science/article/pii/S277237552400131X)
- [A parallel CNN with background removal and lesion segmentation for field plant disease severity classification](https://www.nature.com/articles/s41598-026-51539-2)
- [Effects of segmentation errors on downstream-analysis in highly-multiplexed tissue imaging](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1013350)
- [Orchestrating segment anything models to accelerate segmentation annotation on agricultural image datasets](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1748468/full)
- [An Automated Image Segmentation, Annotation, and Training Framework of Plant Leaves by Joining SAM and YOLOv8](https://doi.org/10.3390/agronomy15051081)
- [Towards precision agriculture: A dataset for early detection of corn leaf pests](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11905841/)
- [Mobile-assisted deep learning framework for identification of insect pests and diseases of maize from field images](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1803005/full)
- [Ultralytics YOLO26: por qué se elimina NMS](https://www.ultralytics.com/blog/why-ultralytics-yolo26-removes-nms-and-how-that-changes-deployment)
- [ultralytics#23685 — yolo26*-seg end2end duplicate detections](https://github.com/ultralytics/ultralytics/issues/23685)
- [ultralytics#6110 — retina_masks y padding no divisible por 2](https://github.com/ultralytics/ultralytics/issues/6110)
- [Ultralytics configuration reference (defaults de `conf` por modo)](https://docs.ultralytics.com/usage/cfg/)
- [Test-Time Augmentation (Ultralytics)](https://docs.ultralytics.com/yolov5/tutorials/test-time-augmentation)
- [Higher accuracy on vision models with EfficientNet-Lite (TensorFlow Blog)](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html)
- [EfficientNet-Lite (tensorflow/tpu)](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)
