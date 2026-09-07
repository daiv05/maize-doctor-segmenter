# DoctorMaiz Leaf Segmentation

Proyecto independiente para preparar, entrenar, auditar y ejecutar un segmentador de
hojas de maíz basado en YOLO instance segmentation. El repositorio no contiene el
clasificador de enfermedades: ese modelo y sus pipelines se trabajan por separado.

## Alcance

- consolidación de fuentes YOLO/COCO y revisión humana;
- normalización JPEG y trazabilidad por SHA-256;
- splits agrupados y reproducibles, sin fugas contra el piloto retenido;
- preflight, empaquetado cloud, entrenamiento, reanudación y evaluación YOLO;
- inferencia del segmentador, selección determinista de la hoja y salidas de máscara;
- métricas IoU/Dice/recall y auditoría del quality gate.

Los datos materializados viven en `data/leaf_detection/`; los checkpoints, paquetes,
predicciones y reportes viven en `outputs/leaf_detection/` y no se versionan.

## Inicio rápido

```bash
python -m venv .venv
source .venv/bin/activate
cp .env.example .env
make install
make leaf-segmentation-status
make leaf-segmentation-verify-locks
make leaf-segmentation-verify-splits
```

El entrenamiento nunca se inicia implícitamente. Los targets de smoke, train y resume
exigen las confirmaciones literales mostradas por `make help`.

## Estructura

```text
cloud_training/   runner y scripts de entrenamiento YOLO en CUDA
config/           configuración de inferencia y quality gate del segmentador
data/             dataset segmentado, piloto, locks y manifiestos (ignorado salvo README)
docs/es/          decisiones, auditorías y flujo técnico de segmentación
scripts/dataset/  consolidación, revisión, finalización y splits
scripts/package/  paquete cloud determinista y verificadores
scripts/pipeline/ preflight y evaluación del segmentador
src/data/         lógica reproducible del dataset de segmentación
src/segmentation/ adaptador YOLO y quality gate
src/preprocessing/geometría, máscara, ROI y letterbox
src/evaluation/   métricas downstream y reliability audit
tests/            pruebas exclusivas del flujo de segmentación
```

## Comandos principales

```bash
make help
make leaf-segmentation-preflight
make leaf-segmentation-downstream-metrics PREDICTIONS=<directorio>
make leaf-segmentation-reliability-audit
make leaf-segmentation-promote-checkpoint

CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 make leaf-segmentation-cloud-smoke
CONFIRM_SEGMENTATION_TRAINING=1 make leaf-segmentation-cloud-train
```

El dataset congelado se publica y se recupera desde Hugging Face:

```bash
make leaf-segmentation-hf-publish HF_SEGMENTATION_STAGE_DIR=<directorio con ~2.4 GB>
make leaf-segmentation-hf-download
```

El entrenamiento remoto siembra ese dataset en un Volume de Modal una sola vez:

```bash
make leaf-segmentation-modal-seed
make leaf-segmentation-modal-verify-dataset
make leaf-segmentation-modal-preflight
CONFIRM_SEGMENTATION_SMOKE_TRAINING=1 make leaf-segmentation-modal-smoke
CONFIRM_SEGMENTATION_TRAINING=1 make leaf-segmentation-modal-train
make leaf-segmentation-modal-promote
make leaf-segmentation-modal-download
```

La documentación de estado y decisiones está en
[`docs/es/leaf-detection/`](docs/es/leaf-detection/segmentation-current-flow.md).
