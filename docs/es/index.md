---
layout: home

hero:
  name: "DoctorMaiz"
  text: "Segmentación de hojas de maíz"
  tagline: "Proyecto independiente para datasets, entrenamiento, inferencia y auditoría de máscaras de hoja."
  image:
    src: /logo.svg
    alt: DoctorMaiz
  actions:
    - theme: brand
      text: Flujo actual
      link: /es/leaf-detection/segmentation-current-flow
    - theme: alt
      text: Entrenamiento cloud
      link: /es/leaf-detection/segmentation-cloud-training

features:
  - title: "Dataset trazable"
    details: "Consolidación YOLO/COCO, revisión humana, normalización JPEG, locks y fingerprints SHA-256."
  - title: "YOLO instance segmentation"
    details: "Entrenamiento reproducible de yolo26n-seg con preflight, smoke, resume y test retenido."
  - title: "Quality gate"
    details: "Selección determinista de hoja, métricas geométricas y auditoría visual contra casos etiquetados."
---

## Alcance

Este repositorio contiene sólo el segmentador de hojas. El clasificador de
enfermedades, plagas o deficiencias y sus artefactos se mantienen en un proyecto
separado.

El flujo activo cubre la preparación del dataset de máscaras, los splits sin fugas,
el entrenamiento remoto, la inferencia del segmentador y las evaluaciones de calidad.

## Rutas principales

- [Flujo actual](/es/leaf-detection/segmentation-current-flow)
- [Fuentes externas](/es/leaf-detection/external-segmentation-datasets-eda)
- [Splits reproducibles](/es/leaf-detection/segmentation-dataset-splits)
- [Preflight](/es/leaf-detection/segmentation-training-preflight)
- [Entrenamiento cloud](/es/leaf-detection/segmentation-cloud-training)
- [Protocolo de mejora](/es/leaf-detection/segmentation-improvement-protocol)
- [Resultados de D-01](/es/leaf-detection/segmentation-d01-results)
- [Quality gate](/es/leaf-detection/segmentation-reliability-gate-audit)
- [Revisión profunda del pipeline](/es/leaf-detection/segmentation-pipeline-deep-review)
- [Decisiones de arquitectura](/es/decisions/adr-leaf-instance-segmentation-strategy)
