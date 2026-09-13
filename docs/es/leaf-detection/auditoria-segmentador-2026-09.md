# Auditoría del segmentador — septiembre 2026

Auditoría de funcionalidades, bugs e inconsistencias del repositorio
`maize-doctor-segmenter`, más la comparación de su lógica de entrenamiento e
inferencia contra `maize-doctor-classifier`, con foco en Modal y su uso de
almacenamiento.

- **Fecha:** 2026-09-06
- **Commit auditado:** `985a920` (`feat: segmenter completed`)
- **Alcance:** `src/`, `scripts/`, `cloud_training/`, `modal_training.py`,
  `Makefile`, `config/segmentation.yaml`, `tests/`.
- **Comparación:** `maize-doctor-classifier` en `ae5d1ed`.

## 1. Resumen ejecutivo

El segmentador es un proyecto con **muy buena higiene de reproducibilidad**:
fingerprints SHA-256 en cada frontera, guards de confirmación literal,
escrituras atómicas (`os.replace` + `fsync`), manifiestos de identidad de run,
gates de un solo uso sobre `test` y contratos estáticos sobre el plano de
control de Modal. La calidad del código de dominio (geometría de máscara,
selección determinista de hoja, splits por grupos) es alta y está bien probada:
**342 de 346 pruebas pasan**.

Los problemas no están en el modelo ni en el dataset, sino en **tres costuras**:

1. **El pipeline está bloqueado en su último gate** por constantes rígidas
   (`183` instancias de test) que ya no coinciden con lo que Ultralytics cuenta
   efectivamente (`182`). Es un bloqueo conocido y documentado, pero sin
   remediación en código.
2. **La cadena de custodia tiene un eslabón sin pin**: los pesos iniciales
   `yolo26n-seg.pt` se descargan de internet en el preflight y su SHA-256 se
   *registra* en lugar de *verificarse* contra un valor esperado. Todo lo demás
   del repositorio está anclado a hashes congelados.
3. **La configuración del runtime está bifurcada**: `config/segmentation.yaml` y
   `src/config.py` gobiernan la inferencia local, pero el camino cloud/Modal
   usa un juego paralelo de variables (`LEAF_SEGMENTATION_*`) que ignora
   `get_output_root()`. De ahí se derivan divergencias reales de umbrales y
   artefactos que se escriben fuera del árbol descargable.

Frente al clasificador, el segmentador es **más riguroso pero mucho menos
operable**: no tiene exportación (ONNX/TFLite), no tiene promoción automática
del checkpoint entrenado al que consume la inferencia, no puede evaluar sus
propios experimentos sobre `test`, y su paquete de código viaja como un tarball
de 2.13 GB subido a mano en vez de construirse desde el árbol de trabajo.

### Tabla de hallazgos

| # | Severidad | Hallazgo | Ubicación |
|---|---|---|---|
| B-01 | **Bloqueante** | Conteo de instancias de test congelado en 183 vs. 182 efectivas | `cloud_training/run_ultralytics.py:47` |
| B-02 | **Bloqueante** | Sin promoción del checkpoint entrenado al que consume la inferencia | `config/segmentation.yaml:7` |
| A-01 | Alta | Umbrales del quality gate en código ≠ los calibrados en YAML | `src/segmentation/quality.py:52-53` |
| A-02 | Alta | `reject_multiple_eligible` con defaults contradictorios | `src/segmentation/quality.py:193,249` |
| A-03 | Alta | Pesos iniciales sin SHA-256 esperado (único eslabón sin pin) | `scripts/pipeline/leaf_segmentation_cloud_preflight.py:192` |
| A-04 | Alta | Los experimentos no pueden evaluarse sobre `test` | `cloud_training/run_ultralytics.py:493` |
| A-05 | Alta | Artefactos de runtime de Modal escritos fuera del árbol descargable | `modal_training.py:389` vs. `Makefile:30` |
| A-06 | Alta | La auditoría de fiabilidad exige el corpus del clasificador | `scripts/experiments/audit_segmentation_reliability.py:293` |
| M-01 | Media | Normalización EXIF inconsistente entre calibración y auditoría | `scripts/experiments/calibrate_segmentation_selection_threshold.py:154` |
| M-02 | Media | `fallback` y `not detected` son el mismo booleano | `src/evaluation/segmentation_downstream.py:139,301` |
| M-03 | Media | `_distance` compara escalas incompatibles en la calibración | `src/evaluation/segmentation_gate_calibration.py:162` |
| M-04 | Media | `FORCE_INTERNAL_TEST_RERUN=1` es una salida muerta | `cloud_training/evaluate_test.sh:12` |
| M-05 | Media | `modal_training.py` fuera de `make lint` y `make check` | `Makefile:352,361` |
| M-06 | Media | `python -m pytest` no pasa en un checkout limpio | `tests/checks/`, `tests/package/` |
| M-07 | Media | `target_size` de handoff hardcodeado en `(224, 224)` | `src/preprocessing/segmented_leaf_processor.py:92,140` |
| M-08 | Media | `torch` importado en el camino "read-only" del preflight | `src/training/segmentation_preflight.py:20` |
| B-03 | Baja | `connected_components` no puede informar nada por construcción | `src/preprocessing/leaf_mask.py:243` |
| B-04 | Baja | Comprobación `np.isfinite` muerta sobre un array `uint8` | `src/preprocessing/leaf_mask.py:266` |
| B-05 | Baja | `max(0.0, ...)` muerto en `over_segmentation_ratio` | `src/evaluation/segmentation_downstream.py:178` |
| B-06 | Baja | `LetterboxResult` mezcla `(alto, ancho)` y `(ancho, alto)` | `src/preprocessing/letterbox.py:18-23` |
| B-07 | Baja | `make fmt` reformatearía 26 archivos | `Makefile:358` |
| B-08 | Baja | `validate` y `test` son el mismo comando con distinta descripción | `cloud_training/validate.sh`, `evaluate_test.sh` |
| B-09 | Baja | `model` del config final congelado no describe la inicialización real | `cloud_training/run_ultralytics.py:724` |

## 2. Método

Lo que se ejecutó realmente en esta auditoría:

```
python -m pytest          -> 342 passed, 4 failed, 3 subtests
python -m ruff check .    -> All checks passed
python -m ruff format --check src/ scripts/  -> 26 files would be reformatted
```

El entorno no tenía el dataset materializado (`data/leaf_detection/` está en
`.gitignore`), por lo que **no se pudo ejecutar** ningún flujo que toque
imágenes reales, GPU, Ultralytics ni Modal. Los hallazgos sobre esos caminos se
derivan de lectura de código y de verificación aislada de las funciones puras
implicadas (ver §3, cada hallazgo indica cómo se comprobó).

Verificaciones puntuales ejecutadas en Python contra el código real del
repositorio: divergencia de umbrales del quality gate, complementariedad de
`detected`/`fallback`, comportamiento de `np.isfinite` sobre `uint8` y conteo de
componentes conexas sobre polígonos cóncavos y anulares.

## 3. Hallazgos

### B-01 · El gate final está bloqueado por un conteo congelado

`cloud_training/run_ultralytics.py:46-47` fija:

```python
EXPECTED_TEST_IMAGE_COUNT = 173
EXPECTED_TEST_INSTANCE_COUNT = 183
```

`validate_test_evaluation_inputs()` cuenta 183 líneas en los `.txt` de
`labels/test`, pero `capture_validation_observation()` lee el conteo *efectivo*
del dataloader de Ultralytics, que deduplica etiquetas por `clase+bbox`. Dos
polígonos distintos que ocupan todo el marco comparten bbox, y el conteo cae a
182. La comparación de `run_ultralytics.py:1006-1015` entonces siempre falla, y
`modal_training.py:1051-1052` repite el mismo `!= 183`.

`docs/es/leaf-detection/segmentation-current-flow.md` ya documenta esto como
"S13 BLOQUEADO" y explica que el runner prefiere bloquear antes que ocultar la
diferencia. La decisión de diseño es correcta; lo que falta es la remediación:
hoy **S13 y S14 (piloto externo) son inalcanzables** y ninguna constante puede
cambiarse sin editar dos archivos a la vez.

**Recomendación.** Separar el conteo de anotaciones (183, propiedad del dataset)
del conteo efectivo del cargador (182, propiedad de Ultralytics), declarar ambos
como constantes distintas con su justificación, y derivarlos de un solo lugar
—preferiblemente del `split_lock.json`— en vez de duplicarlos entre el runner y
el plano de control.

### B-02 · No existe promoción del checkpoint entrenado

`config/segmentation.yaml:7` declara el checkpoint de inferencia:

```yaml
checkpoint: "leaf_detection/models/doctor_maiz_leaf_segmenter_best.pt"
```

Se resuelve bajo `get_output_root()`. Pero el entrenamiento escribe en
`outputs/leaf_detection/segmenter/yolo26n_seg_baseline/weights/best.pt`. **No
hay ningún script, target de `make` ni función de Modal que copie uno al otro**,
ni manifiesto de procedencia para el archivo promovido — a pesar de que el
repositorio sí tiene `register_verified_experiment_weights.py` para los pesos
`yolo26s`, con verificación de hash y negativa a sobrescribir.

Consecuencia: `make leaf-segmentation-reliability-audit` y
`make leaf-segmentation-calibrate-selection` dependen de un archivo que sólo
existe si alguien lo copió a mano, sin registro de qué run lo produjo.

El clasificador resuelve exactamente este problema con
`scripts/pipeline/sync_mobile_model.py`: lee el `export_summary.json`, copia,
re-calcula el hash de la copia, aborta si no coincide y deja un `manifest.json`
de procedencia. **Es el patrón a portar.**

### A-01 · Los umbrales por defecto del quality gate no son los calibrados

Verificado ejecutando el código:

```
YAML : large_mask_area_ratio=0.25  min_large_mask_bbox_ratio=0.80
CODE : large_mask_area_ratio=0.50  min_large_mask_bbox_ratio=0.70
```

`SegmentationQualityGateConfig` (`src/segmentation/quality.py:51-53`) trae los
valores *pre-calibración*. El comentario del YAML dice explícitamente que 0.25 y
0.80 salieron de la revisión humana de 42 imágenes
(`quality_gate_calibration_v1/summary.json`). Cualquier consumidor que construya
`SegmentationQualityGateConfig()` sin pasar por `from_mapping()` obtiene un gate
**dos veces más permisivo** en área y más laxo en compacidad, en silencio.

`assess_segmentation()` (`quality.py:246-252`) acepta `quality_gate=None` y en
ese caso instancia justamente esos defaults.

**Recomendación.** O bien alinear los defaults del dataclass con la calibración,
o bien eliminar los defaults por completo y hacer obligatorio el paso por
`from_mapping()`, de modo que sea imposible evaluar con un gate no calibrado.

### A-02 · `reject_multiple_eligible` tiene tres verdades

| Fuente | Valor |
|---|---|
| `config/segmentation.yaml` → `quality_gate` | `false` |
| `assess_segmentation()` / `assess_segmentation_legacy()` | `True` |
| `calibrated_status()` / `evaluate_gate()` | `False` |

La calibración (`segmentation_gate_calibration.py:85,119`) replica la política
de producción con `False`, que coincide con el YAML; la API pública de
producción usa `True`. Es decir, el gate calibrado y el gate por defecto de la
librería **no son el mismo gate**, y la diferencia (rechazar toda imagen con más
de una hoja elegible vs. exigir margen de score ≥ 0.33) es sustancial para el
caso real de campo. Mismo remedio que A-01: default único, o sin default.

### A-03 · Los pesos iniciales son el único eslabón sin hash esperado

`leaf_segmentation_cloud_preflight.py:190-193` declara literalmente
`"download_allowed_in_cloud": True` y construye `YOLO("yolo26n-seg.pt")`, lo que
hace que Ultralytics **descargue los pesos de internet**. Luego escribe
`weights_manifest.json` con el SHA-256 de lo que se haya bajado ese día.

`run_ultralytics.py:140-155` (`verified_weights()`) re-calcula ese hash y lo
compara… contra el manifiesto que el propio preflight acaba de escribir. Es una
comprobación de integridad de disco, no de procedencia: **nunca se compara
contra un valor esperado y congelado**.

Contrasta con el resto del repositorio, donde todo tiene su constante
`EXPECTED_*`: fingerprint del padre, fingerprint de test, SHA del paquete, SHA
de `best.pt`, versiones exactas de torch/torchvision/ultralytics/faster-coco-eval
y hasta el digest de la imagen base. Además el manifiesto se sobrescribe en cada
preflight, así que un cambio silencioso aguas arriba no deja rastro.

Nota menor: si el preflight queda bloqueado, `weights_manifest.json` se escribe
igualmente con `resolved: False` y sin clave `path`; `verified_weights()`
fallaría entonces con un `KeyError` en vez de un mensaje accionable.

**Recomendación.** Añadir `EXPECTED_YOLO26N_SEG_SHA256` y verificar contra él,
igual que se hace con `yolo26s` vía `register_verified_experiment_weights.py`.
Idealmente, incluir los pesos base en el paquete cloud y prohibir la descarga.

### A-04 · Los experimentos se entrenan pero no se pueden evaluar

`modal_training.py:113-121` permite siete perfiles de experimento
(`TRAINABLE_EXPERIMENTS`) y expone `experiment` / `resume_experiment`. Pero el
único camino de evaluación, `evaluate_mode()`, pasa por
`validate_test_evaluation_inputs()`, que en `run_ultralytics.py:493-500` exige:

```python
expected_checkpoint = (OUTPUTS / "segmenter/yolo26n_seg_baseline/weights/best.pt")
... and checkpoint_before["sha256"] != EXPECTED_BEST_CHECKPOINT_SHA256
```

Es decir, **sólo el `best.pt` exacto del baseline es evaluable sobre `test`**.
D-01 (`d01_mosaic0_seed42`), que la documentación describe como el candidato de
mejora actual con mAP50-95(M) 0.944, no tiene ninguna ruta a la evaluación final
dentro del runner. Los mismos `validate.sh` y `evaluate_test.sh` apuntan a la
ruta fija del baseline.

Esto es coherente con la política de "test de un solo uso" —que es correcta—,
pero significa que el protocolo de mejora no puede cerrarse: se pueden entrenar
alternativas y no se puede decidir entre ellas con el mismo rigor.

**Recomendación.** Parametrizar el gate de evaluación por *identidad de run*
(manifiesto + SHA registrado en el manifiesto de ese run) en vez de por una
constante única, manteniendo el gate de un solo uso por run.

### A-05 · Artefactos de runtime escritos fuera del árbol descargable

`modal_training.py` mantiene deliberadamente dos raíces en el Volume:

```python
PROJECT_ROOT          = /workspace/project_v7-segmentation-improvements-7a4a5c08-seed42
ARTIFACT_PROJECT_ROOT = /workspace/project_v4-7a4a5c08-seed42
SEGMENTATION_OUTPUT_ROOT = ARTIFACT_PROJECT_ROOT / outputs / leaf_detection
```

El código ejecutable vive en v7 y los artefactos en v4, para conservar la
continuidad de los resultados del baseline. Pero `_runtime_report()`
(`modal_training.py:389`) escribe en **`PROJECT_ROOT`**, no en
`SEGMENTATION_OUTPUT_ROOT`:

```python
runtime_root = PROJECT_ROOT / "outputs" / "leaf_detection" / "modal_runtime"
```

Y `make leaf-segmentation-modal-download` (`Makefile:30`) descarga
`/project_v4-7a4a5c08-seed42/outputs/leaf_detection/`.

Consecuencia concreta: `modal_runtime/*.json`, `pip_freeze.txt`,
`image_build_pip_freeze.txt` y los dos `cloud_training/runtime_environment*.lock`
—precisamente la evidencia de qué entorno y qué GPU ejecutaron cada corrida—
**nunca se descargan**. Es el tipo de artefacto que más se echa de menos cuando
hay que reproducir un resultado meses después.

### A-06 · La auditoría de fiabilidad rompe el alcance del repositorio

`CLAUDE.md` y `README.md` afirman que este repositorio no contiene datos ni
modelos del clasificador. `LOCAL.md` dice que `DATASET_ROOT` "sólo se requiere
para utilidades históricas del piloto".

Sin embargo `audit_segmentation_reliability.py:291-295` —el respaldo de
`make leaf-segmentation-reliability-audit`, listado en el README como comando
principal— hace:

```python
cases = read_audit_manifest(args.manifest, dataset_root=get_dataset_root(),
                            raw_dir=str(paths["raw_dir"]))   # raw_dir: "clean"
```

Es decir, exige `DATASET_ROOT/clean/<clase>/<entorno>/<archivo>`: el corpus de
19 GB del **clasificador**. Un comando de primera línea del segmentador no puede
ejecutarse sin el dataset del otro proyecto, y ni el README ni `LOCAL.md` lo
advierten.

**Recomendación.** O bien materializar las 42 imágenes auditadas dentro de
`data/leaf_detection/` con su lock (son pocas y ya están congeladas), o bien
documentar explícitamente la dependencia y marcarla como opcional en `make help`.

### M-01 · Normalización EXIF inconsistente entre calibración y auditoría

`src/data/loader.py` existe precisamente para garantizar `exif_transpose` + RGB.
La auditoría de fiabilidad lo usa
(`audit_segmentation_reliability.py:312`), pero la calibración del umbral de
selección no:

```python
# calibrate_segmentation_selection_threshold.py:154
with Image.open(image_path) as source:
    image = source.convert("RGB")
```

Ambos scripts alimentan al **mismo** `UltralyticsLeafSegmenter`. En imágenes con
orientación EXIF (típicas de smartphone, que es justo el subconjunto `real`), la
calibración ve la imagen rotada respecto a la auditoría, así que el umbral
elegido no describe exactamente el sistema auditado.

El clasificador declara `load_and_normalize_image()` como "único punto de entrada
a imagen" en su `CLAUDE.md`. El segmentador no tiene esa regla escrita, y el
resultado es esta divergencia. Otros ocho módulos usan `Image.open` directo
(`segmentation_audit.py`, `segmentation_consolidation.py`,
`segmentation_split.py`, `jpeg_normalization.py`, …), pero ahí es correcto:
operan sobre bytes originales para hashear o normalizar, no para inferir.

### M-02 · `fallback` y `not detected` son el mismo booleano

`segmentation_downstream.py:301` calcula `fallback = prediction.sum() /
prediction.size < minimum_area` y `:139` calcula
`detected = predicted_area / total_pixels >= minimum_area`, sobre exactamente los
mismos píxeles y el mismo umbral. Verificado empíricamente: son complementarios
para toda entrada.

Por lo tanto el resumen reporta `fallback_rate` e
`images_without_detection_rate` como si fueran dos señales distintas, cuando son
el mismo número por construcción. Quien lea un `downstream_summary.json` creerá
que dos comprobaciones independientes coinciden.

**Recomendación.** O eliminar una de las dos, o hacer que `fallback` refleje lo
que su nombre promete: que el `SegmentedLeafProcessor` cayó a
`FALLBACK_ORIGINAL`/`FALLBACK_REJECT` (información que existe en
`SegmentedLeafProcessingResult.fallback_used` y que aquí no se está usando).

### M-03 · La calibración del gate compara escalas incompatibles

`segmentation_gate_calibration.py:162-168`:

```python
return sum(abs(left[key] - right[key]) for key in left)
```

Suma diferencias absolutas sobre cinco umbrales, cuatro de los cuales viven en
`[0, 1]` y uno —`max_large_mask_normalized_perimeter`— en la rejilla
`(6.0, 7.0, 8.0, 9.0)`. Una diferencia de 1.0 en el perímetro pesa lo mismo que
recorrer *entero* el rango de los otros cuatro juntos.

`_distance` sólo se usa como último criterio de desempate para elegir "el gate
menos disruptivo", así que el impacto está acotado, pero el desempate está
gobernado casi exclusivamente por la dimensión del perímetro. Normalizar cada
término por el rango de su rejilla lo arregla.

Nota adicional: `calibrated_status()` usa `assert` para descartar `None`
(`segmentation_gate_calibration.py:104-106`); bajo `python -O` esas comprobaciones desaparecen
y el fallo se convierte en un `TypeError` río abajo.

### M-04 · `FORCE_INTERNAL_TEST_RERUN=1` no puede funcionar

`evaluate_test.sh:12-17` ofrece una salida de emergencia documentada:

```bash
if [[ -f "${TEST_SUMMARY}" && "${FORCE_INTERNAL_TEST_RERUN:-0}" != "1" ]]; then
  ...
  echo "Solo con una decision formal registrada: FORCE_INTERNAL_TEST_RERUN=1" >&2
```

Pero aunque se supere ese guard, `validate_test_evaluation_inputs()`
(`run_ultralytics.py:483-492`) aborta si existe cualquiera de
`yolo26n_seg_test/`, `yolo26n_seg_test_predictions/` o `test_summary.json` —y si
ya hubo una evaluación, existen los tres. La variable promete una vía que no
existe: el operador acabará borrando artefactos a mano, que es justo lo que el
gate quería evitar.

### M-05 · `modal_training.py` no pasa por lint ni por type-check

`Makefile:352,361`:

```make
lint:   $(RUFF) check src/ scripts/
check:  $(PYRIGHT) src/ scripts/
```

`modal_training.py` (38 KB, todo el plano de control de Modal) y `tests/` quedan
fuera de ambos. `ruff check .` sí pasa hoy, pero por casualidad: nada lo
garantiza en el futuro y `pyright` nunca lo ha visto. Su única red de seguridad
son los contratos de `tests/checks/test_modal_training_contract.py`, que
comparan **cadenas literales del código fuente** (`assertIn('VOLUME_MOUNT =
Path("/workspace")', SOURCE)`) — un enfoque deliberado y útil contra cambios
accidentales, pero que no detecta ningún error de tipos.

### M-06 · `python -m pytest` no pasa en un checkout limpio

`CLAUDE.md` documenta `python -m pytest` como el comando de verificación. En un
clon fresco falla:

```
FAILED tests/checks/test_makefile_safety.py::test_status_is_read_only
FAILED tests/package/test_leaf_segmentation_cloud_package.py::test_cloud_payload_fingerprints_are_valid_without_all_tree
FAILED tests/package/test_leaf_segmentation_cloud_package.py::test_allow_list_excludes_protected_and_historical_trees
FAILED tests/package/test_leaf_segmentation_cloud_package.py::test_pilot_transport_manifest_is_separate
4 failed, 342 passed, 3 subtests passed
```

Las cuatro requieren `data/leaf_detection/detector_dataset/` materializado, que
está en `.gitignore`. Fallan con `FileNotFoundError` en lugar de saltarse con
`pytest.mark.skipif`, así que la suite no distingue "el dataset no está" de "el
código se rompió".

Además, sin el extra `[segmentation]` la suite ni siquiera colecta: tres módulos
mueren con `ModuleNotFoundError: torch` (ver M-08).

### M-07 · El `target_size` del handoff está hardcodeado

`segmented_leaf_processor.py:92` y `:140` fijan `(224, 224)` en el código, y
`config/segmentation.yaml` **no tiene** clave `target_size`. Ese es el tamaño con
el que el perfil `CROP_MASK_LETTERBOX` entrega el recorte al clasificador, cuyo
`config/dataset.yaml:6` sí lo declara (`target_size: [224, 224]`).

Es decir, el contrato de tamaño entre los dos modelos está declarado en un
repositorio y hardcodeado en el otro. Si el clasificador migra a 256 o a
resoluciones por modelo (`src/models/input_sizes.py` ya lo contempla), el
segmentador seguirá entregando 224 sin que nada falle.

### M-08 · `torch` en el camino "read-only"

`src/training/segmentation_preflight.py:20` importa `torch` a nivel de módulo,
aunque sólo lo usa para el reporte de entorno y el forward sintético
(líneas 312-346, 562-619). Ese módulo lo importa
`scripts/package/leaf_segmentation_make.py:14`, que respalda
`make leaf-segmentation-status`, `verify-locks`, `verify-splits`,
`package-verify`, `results` y `checksums`: **todos los targets de la sección
"LOCAL / SEGURO" del `make help` exigen la pila CUDA completa**.

Contrasta con `src/segmentation/leaf_segmenter.py`, que documenta e implementa
justamente la política opuesta ("importa Ultralytics sólo cuando se pide
inferencia real"). Mover `import torch` dentro de las dos funciones que lo
necesitan hace ejecutables los comandos seguros con sólo el extra `dev`.

### Hallazgos menores

- **B-03 · `connected_components` no puede informar nada.**
  `mask_geometry()` mide componentes conexas y `largest_component_ratio`, y esos
  campos alimentan `AUDIT_NUMERIC_FIELDS` de la auditoría. Pero la máscara que
  reciben proviene siempre de `rasterize_instance_polygon()`, que rellena **un
  solo polígono simple**. Verificado con un polígono cóncavo en U y con un anillo:
  ambos dan `connected_components=1`, `largest_component_ratio=1.0`. La métrica
  sólo podría superar 1 con polígonos auto-intersectantes patológicos. Además,
  como Ultralytics entrega un contorno por instancia en `masks.xy`, una hoja
  partida por oclusión pierde el fragmento menor antes de llegar aquí.
- **B-04 · Comprobación muerta.** `leaf_mask.py:266` valida
  `np.isfinite(output).all()` sobre un array `uint8`: siempre `True` (verificado).
- **B-05 · `max()` muerto.** `segmentation_downstream.py:178`:
  `max(0.0, predicted_area - intersection)`; la intersección nunca excede el área
  predicha.
- **B-06 · Convención mezclada en `LetterboxResult`.** `original_size` y
  `resized_size` son `(ancho, alto)` de Pillow; `target_size` es `(alto, ancho)`
  por convención del proyecto. Ambas conviven en el mismo dataclass sin marca.
  Con `(224, 224)` el error es invisible; con un tamaño no cuadrado no lo será.
- **B-07 · `make fmt` es una bomba de diff.** `ruff format --check` reporta 26 de
  55 archivos pendientes. `make check` no incluye `format --check`, así que la
  deriva crece en silencio y el primer `make fmt` producirá un diff enorme
  mezclado con cambios reales.
- **B-08 · `validate` y `test` son el mismo comando.** `validate.sh` y
  `evaluate_test.sh` invocan `run_ultralytics.py evaluate` con el mismo
  checkpoint, config, split y directorio de salida. `make help` los describe como
  "test retenido" y "test interno", sugiriendo dos conjuntos distintos que no
  existen. Sólo difieren en el guard.
- **B-09 · El config final congelado miente sobre el modelo.**
  `run_ultralytics.py:724` escribe `final["model"] = MODEL_NAME`
  (`"yolo26n-seg.pt"`) en `train_yolo26n_seg.final.yaml`. En la corrida real ese
  valor se descarta: `resolve_initial_model()` lo sobrescribe con la ruta de
  `weights_manifest.json`. El artefacto que se conserva como registro de la
  decisión no describe la inicialización que efectivamente ocurrió.
- **Duplicación transversal.** `sha256()` está reimplementado en al menos cinco
  módulos (`leaf_segmenter.py`, `run_ultralytics.py`, `modal_training.py`,
  `leaf_segmentation_cloud_preflight.py`, `register_verified_experiment_weights.py`)
  y `project_path()` en tres. `src/data/leaf_pilot.py` ya expone `sha256_file()`.

## 4. Comparación con el clasificador

### 4.1 Vista general

| Dimensión | `maize-doctor-classifier` | `maize-doctor-segmenter` |
|---|---|---|
| Framework | PyTorch + timm, bucle propio (`src/training/loop.py`) | Ultralytics YOLO, bucle delegado |
| Entrada del entrenamiento | `outputs/splits/seed_42/*.csv` | `data/.../detector_dataset/dataset.yaml` |
| Config de dominio | `config/dataset.yaml` (clases, `target_size`, seed) | `config/segmentation.yaml` (sólo inferencia) + YAML de Ultralytics |
| Resolución de rutas | `get_dataset_root()` / `get_output_root()` | idem **en local**; `LEAF_SEGMENTATION_*` en cloud |
| Identidad de run | `<modelo>/<run_id timestamp>/` + `latest.json` | `segmenter/<name>/` fijo + `active_run_manifest.json` |
| Múltiples runs | naturales, versionados por timestamp | prohibidos (`exist_ok=False`, aborta si existe) |
| Reanudación | no | sí, con manifiesto e histórico timestamped |
| Guards de confirmación | ninguno | `CONFIRM_*=1` literal en smoke/train/resume/piloto/clean |
| Fingerprints de datos | SHA-256 para dedup en `create_splits.py` | fingerprints congelados verificados en cada gate |
| Evaluación final | `test` en cada run, libre | `test` de un solo uso, con gate de colisión |
| Exportación | ONNX + TFLite, int8, paridad numérica, `labels.json` | **inexistente** |
| Evaluación del exportado | `evaluate_export.py` sobre test completo | n/a |
| Promoción a la app | `sync_mobile_model.py` con verificación de hash | **inexistente** (ver B-02) |
| Explicabilidad | LIME, SHAP, Grad-CAM, fidelidad, errores | quality gate + auditoría visual humana |
| OOD | Mahalanobis relativa (`compute_ood_stats.py`) | n/a |
| Lint / types | `src/ scripts/` | `src/ scripts/` (deja fuera `modal_training.py`) |

Lectura corta: el clasificador está optimizado para **iterar y desplegar**; el
segmentador para **no poder equivocarse**. Ninguno de los dos extremos es
gratuito. El segmentador no puede producir un artefacto desplegable; el
clasificador no puede demostrar que su `test` no se ha contaminado.

### 4.2 Modal: dos arquitecturas distintas

#### Cómo llega el código al contenedor

**Clasificador** — el código viaja *en la imagen*:

```python
# scripts/modal/_common.py
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "torchvision==0.27.1",
                 index_url="https://download.pytorch.org/whl/cu126")
    .pip_install_from_pyproject("pyproject.toml",
                                optional_dependencies=["cloud", "xai", "export"])
    .env({"DATASET_ROOT": "/data", "OUTPUT_ROOT": "/outputs", ...})
    .add_local_dir("config", "/root/config", copy=True)
    .add_local_python_source("src", "scripts"))
```

Cada `modal run` sincroniza el árbol de trabajo. Cambiar una línea de `train.py`
y lanzarlo en GPU es un solo comando.

**Segmentador** — el código viaja *en el Volume*, como release congelada:

```python
PACKAGE_NAME   = "doctor_maiz_leaf_segmentation_cloud_v7-...-seed42.tar.gz"  # 2.13 GB
PACKAGE_SHA256 = "a90f3f30...9655b6"
```

Flujo: `make leaf-segmentation-cloud-package` → `…-modal-upload` (verifica SHA
local y sidecar) → `…-modal-prepare` (verifica SHA remoto, extrae con
`filter="data"`, rechaza symlinks/rutas absolutas/`..`, valida
`package_manifest.json`, corre `sha256sum --check`, escaneo JPEG/Ultralytics,
`rename()` atómico, marcador `.modal_package_prepared.json`). El código se
alcanza inyectando `PROJECT_ROOT` en `PYTHONPATH`, sin instalación editable.

El paquete incluye el dataset, así que **no hay descarga remota de datos**: el
segmentador no necesita ni HF Hub ni secretos en Modal. El precio es un ciclo de
iteración de 2.13 GB por cambio de código.

#### Volúmenes y layout

| | Clasificador | Segmentador |
|---|---|---|
| Volúmenes | 2: `corn-clean` → `/data`, `corn-outputs` → `/outputs` | 1: `doctor-maiz-leaf-segmentation` → `/workspace` |
| `create_if_missing` | `True` | `False` (falla si no existe; hay target `modal-volume-create`) |
| Separación datos/artefactos | por volumen | por subárbol dentro del mismo volumen |
| Poblado de datos | `seed_dataset()` remoto desde HF Hub, con `--force` | dentro del tarball |
| Actualizar datos | `make modal-seed FORCE=1 && make modal-splits` | reconstruir y resubir el paquete completo |
| Descarga de resultados | `make modal-pull` → `modal volume get --force corn-outputs / ./outputs-remote` | `modal volume get --force <vol> /project_v4-…/outputs/leaf_detection/ <dir>` |
| Limpieza | `clean_outputs()` remoto | ninguna (`make clean-outputs` es local, con `CONFIRM_CLEAN_OUTPUTS=1`) |
| Secretos | `modal.Secret.from_name("hf")` en todas las funciones | ninguno |
| `reload()` / `commit()` | `reload()` antes de leer, `commit()` al final de cada función | `_reload_workspace_before_access()` (con `chdir /tmp`) y `commit()` en `finally` |

El manejo de consistencia del segmentador es **más correcto**: el `commit()` en
`finally` (`modal_training.py:547-548`) persiste los artefactos incluso cuando la
corrida falla, y el `os.chdir("/tmp")` antes de `reload()` evita quedarse con un
CWD sobre un inode reemplazado. El clasificador sólo hace `commit()` en el camino
feliz: **si `train_main` muere a mitad, se pierde el run**. Es la práctica que
conviene copiar en dirección inversa.

#### Configuración del runtime

El clasificador inyecta `DATASET_ROOT` y `OUTPUT_ROOT` en `.env()` de la imagen,
de modo que `get_dataset_root()` / `get_output_root()` resuelven a los mounts sin
que ningún script sepa que está en Modal. **Una sola vía de configuración,
local y remota.**

El segmentador tiene `get_output_root()` y `get_project_data_root()` en
`src/config.py`, pero el camino cloud no los usa: `run_ultralytics.py:31-40` y
`leaf_segmentation_cloud_preflight.py:25-34` reimplementan `project_path()` sobre
`LEAF_SEGMENTATION_OUTPUT` / `LEAF_SEGMENTATION_DATASET`, y `modal_training.py`
los pasa como variables de `make` y de entorno. Resultado: dos jerarquías de
configuración que hay que mantener sincronizadas a mano, y de las que se derivan
A-05 (artefactos fuera del árbol descargable) y parte de la confusión v4/v7.

#### Validación de entorno

Aquí el segmentador está muy por delante y **el clasificador debería copiarlo**:

- Imagen base anclada por digest (`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f…`).
- Todas las versiones pip pinneadas exactas, `pip check` en build.
- `IMAGE_RECIPE_SHA256`: hash de la receta completa de la imagen.
- `.run_function(_validate_modal_image_versions)` valida python/torch/torchvision/
  ultralytics/faster-coco-eval **en tiempo de build**.
- `_runtime_report()` revalida en cada corrida y además comprueba GPU real:
  nombre coincidente con `DOCTOR_MAIZ_MODAL_GPU`, VRAM ≥ 12 GiB, `nvidia-smi`.
- `installed_distribution_snapshot()` hashea el inventario de paquetes antes y
  después de evaluar, para detectar instalaciones en caliente.

El clasificador no valida nada de esto: si Modal sirve una GPU distinta o pip
resuelve otra versión, el run continúa.

#### Guards

| Operación | Clasificador | Segmentador |
|---|---|---|
| Entrenar | `make modal-train` | `CONFIRM_SEGMENTATION_TRAINING=1` (make) **+** `--confirm true` (Modal) **+** `CONFIRM_…=1` (shell) **+** validación del config congelado |
| Borrar outputs remotos | `make modal-clean-outputs`, sin confirmación | no existe |
| Repetir `test` | libre | gate de colisión de artefactos |

`make modal-clean-outputs` del clasificador borra todo el Volume `corn-outputs`
—splits, runs, checkpoints, exportaciones— **sin ninguna confirmación**. Es el
riesgo operativo más alto de los dos repositorios y el patrón
`CONFIRM_CLEAN_OUTPUTS=1` del segmentador lo resuelve en una línea.

### 4.3 Inferencia

| | Clasificador | Segmentador |
|---|---|---|
| Entrada | `load_and_normalize_image()`, punto único declarado | `image_to_rgb()`; `load_and_normalize_image()` sólo en algunos scripts (M-01) |
| Preproceso | resize/normalización del pipeline de datos | selección determinista de instancia + 4 perfiles de máscara |
| Salida | clase + confianza + OOD | máscara + bbox + traza por instancia + estado del gate |
| Trazabilidad | `predictions.csv` | `metadata.json` por imagen, con score de cada instancia y motivo de rechazo |
| Artefacto desplegable | `model.tflite` + `labels.json` + `ood_stats.json` | ninguno |
| Versión del runtime | flexible | `ultralytics == 8.4.104` exacta, o falla al cargar |

La traza de decisión del segmentador (`InstanceSelectionTrace` por instancia, con
`relative_area`, `center_proximity`, `score` y motivo textual de rechazo) es
**mejor instrumentación que la del clasificador**, que sólo persiste predicción y
confianza. Vale la pena portar esa idea.

El bloqueo grande es la ausencia de exportación: no hay ONNX, ni TFLite, ni
paridad numérica, ni `labels.json`. Para un flujo móvil donde el segmentador
corre *antes* del clasificador, hoy no existe manera de empaquetarlo.

## 5. Plan de convergencia

Orden propuesto, de mayor a menor retorno.

### Fase 1 — Desbloquear (B-01, B-02, A-04)

1. Separar `EXPECTED_TEST_ANNOTATION_COUNT` (183) de
   `EXPECTED_TEST_EFFECTIVE_COUNT` (182), derivarlos del lock y dejar de
   duplicarlos entre `run_ultralytics.py` y `modal_training.py`. Desbloquea S13
   y S14.
2. Añadir `scripts/pipeline/promote_segmenter_checkpoint.py` calcado de
   `sync_mobile_model.py` del clasificador: copia, re-hashea la copia, aborta si
   difiere, escribe `manifest.json` de procedencia. Exponerlo como
   `make leaf-segmentation-promote-checkpoint RUN=<run_id>`.
3. Parametrizar el gate de evaluación por identidad de run en vez de por
   `EXPECTED_BEST_CHECKPOINT_SHA256`, conservando el gate de un solo uso por run.

### Fase 2 — Unificar configuración (A-01, A-02, A-05, M-07)

4. Un único origen para los umbrales del quality gate: eliminar los defaults del
   dataclass o alinearlos con la calibración, y unificar
   `reject_multiple_eligible`.
5. Adoptar el patrón del clasificador en Modal: inyectar `OUTPUT_ROOT` y
   `PROJECT_DATA_ROOT` en `.env()` de la imagen y hacer que `run_ultralytics.py`
   y el preflight usen `get_output_root()` / `get_project_data_root()`. Elimina
   `project_path()` duplicado y la deriva v4/v7 que causa A-05.
6. Mientras tanto, corregir `runtime_root` para que apunte a
   `SEGMENTATION_OUTPUT_ROOT`.
7. Declarar `target_size` en `config/segmentation.yaml` y documentarlo como el
   contrato de handoff con el clasificador.

### Fase 3 — Cerrar la cadena de custodia (A-03, A-06)

8. `EXPECTED_YOLO26N_SEG_SHA256` verificado en el preflight, o pesos base dentro
   del paquete cloud.
9. Materializar las 42 imágenes de la auditoría de fiabilidad dentro de
   `data/leaf_detection/` (con lock), eliminando la dependencia de
   `DATASET_ROOT` del clasificador. Si no es viable, documentarla en `LOCAL.md` y
   `make help`.

### Fase 4 — Paridad operativa (exportación)

10. Portar `src/export/` al segmentador para YOLO-seg: ONNX de un solo archivo,
    TFLite, validación de paridad numérica sobre una muestra del test,
    `export_summary.json`. Ultralytics tiene `model.export()`, pero la envoltura
    de paridad y consolidación del clasificador es lo que aporta valor.
11. Equivalente de `evaluate_export.py`: correr el archivo exportado sobre el
    split completo y reportar la caída real de IoU/Dice, no sólo la paridad
    numérica sobre ~30 muestras.

### Fase 5 — Higiene (M-04 a M-08, menores)

12. Incluir `modal_training.py` y `tests/` en `make lint` y `make check`.
13. `pytest.mark.skipif` en las cuatro pruebas dependientes del dataset, para que
    un checkout limpio distinga "falta el dataset" de "está roto".
14. Mover `import torch` dentro de las funciones que lo usan en
    `segmentation_preflight.py`.
15. Aplicar `ruff format` en un commit aislado y añadir `format --check` a
    `make check`.
16. Eliminar `FORCE_INTERNAL_TEST_RERUN` o hacerlo funcional; fusionar
    `validate.sh` y `evaluate_test.sh`, o diferenciarlos de verdad.
17. Correcciones puntuales: `_distance` normalizado por rango, `fallback`
    derivado de `fallback_used`, retirar comprobaciones muertas (`np.isfinite`
    sobre `uint8`, `max(0.0, …)`), retirar `connected_components` de la auditoría
    o alimentarlo con la máscara sin rasterizar, unificar la convención de
    `LetterboxResult`, centralizar `sha256()` en `src/`.

### En dirección inversa: qué debe copiar el clasificador

- `commit()` del Volume en `finally`, no sólo en el camino feliz.
- Confirmación literal para `modal-clean-outputs`.
- Anclaje de la imagen base por digest y validación de versiones en build.
- Verificación de la GPU efectivamente asignada.
- Traza de decisión por instancia al estilo `InstanceSelectionTrace`.

## 6. Anexos

### A. Salida de las comprobaciones ejecutadas

```
$ python -m pytest -q
4 failed, 342 passed, 3 subtests passed in 7.70s

$ python -m ruff check .
All checks passed!

$ python -m ruff format --check src/ scripts/
26 files would be reformatted, 29 files already formatted
```

### B. Verificaciones puntuales

```
Umbrales del quality gate
  YAML : large_mask_area_ratio=0.25  min_large_mask_bbox_ratio=0.80
  CODE : large_mask_area_ratio=0.50  min_large_mask_bbox_ratio=0.70

apply_leaf_mask: np.isfinite sobre uint8 -> True siempre

detected / fallback (umbral 0.01)
  0.000  detected=False  fallback=True   complementarios
  0.005  detected=False  fallback=True   complementarios
  0.010  detected=True   fallback=False  complementarios
  0.500  detected=True   fallback=False  complementarios

mask_geometry sobre polígono rasterizado
  cóncavo en U : connected_components=1  largest_component_ratio=1.0
  anillo       : connected_components=1  mask_area_ratio=0.504
```

### C. Limitaciones de esta auditoría

- Sin dataset materializado: no se ejecutó ningún flujo sobre imágenes reales.
- Sin GPU ni Ultralytics instalado: `leaf_segmenter.py`, `run_ultralytics.py` y
  el preflight cloud se auditaron por lectura, no por ejecución.
- Sin acceso a Modal: el plano de control se auditó por lectura y contra sus
  pruebas de contrato.
- No se auditaron en profundidad los módulos de preparación de datos
  (`segmentation_audit.py`, `segmentation_consolidation.py`,
  `segmentation_split.py`, `roi_manifest.py`, ~7 900 líneas). Su lógica de
  determinismo y anti-fugas se revisó en superficie y no se hallaron problemas,
  pero merecen una pasada dedicada.
