import { defineConfig } from "vitepress";

const segmentationSidebar = [
  {
    text: "Segmentación",
    items: [
      { text: "Flujo actual", link: "/es/leaf-detection/segmentation-current-flow" },
      { text: "Historia técnica", link: "/es/leaf-detection/history" },
      { text: "Fuentes externas", link: "/es/leaf-detection/external-segmentation-datasets-eda" },
      { text: "Dataset YOLO", link: "/es/leaf-detection/yolo26-detector-dataset" },
      { text: "Splits", link: "/es/leaf-detection/segmentation-dataset-splits" },
      { text: "Preflight", link: "/es/leaf-detection/segmentation-training-preflight" },
      { text: "Entrenamiento cloud", link: "/es/leaf-detection/segmentation-cloud-training" },
      { text: "Mejoras", link: "/es/leaf-detection/segmentation-improvement-protocol" },
      { text: "Resultados D-01", link: "/es/leaf-detection/segmentation-d01-results" },
      { text: "Quality gate", link: "/es/leaf-detection/segmentation-reliability-gate-audit" },
      { text: "Revisión profunda", link: "/es/leaf-detection/segmentation-pipeline-deep-review" },
    ],
  },
  {
    text: "Decisiones",
    items: [
      { text: "Estrategia de instancias", link: "/es/decisions/adr-leaf-instance-segmentation-strategy" },
      { text: "Fuentes externas", link: "/es/decisions/adr-external-leaf-segmentation-datasets" },
      { text: "Entrenamiento", link: "/es/decisions/adr-segmentation-training-strategy" },
      { text: "Datos y outputs", link: "/es/decisions/adr-project-data-root-and-output-root" },
    ],
  },
];

export default defineConfig({
  vite: { publicDir: "../public" },
  head: [["meta", { name: "robots", content: "noindex, nofollow" }]],
  markdown: { math: true },
  lastUpdated: true,
  locales: {
    es: {
      label: "Español",
      lang: "es-SV",
      link: "/es/",
      title: "DoctorMaiz · Segmentación",
      description: "Segmentación de hojas de maíz con YOLO instance segmentation",
      head: [
        ["meta", { name: "description", content: "Dataset, entrenamiento e inferencia para segmentación de hojas de maíz" }],
        ["meta", { name: "keywords", content: "maíz, hoja, segmentación, YOLO, máscaras" }],
        ["link", { rel: "icon", type: "image/svg+xml", href: "/logo.svg" }],
      ],
      themeConfig: {
        nav: [
          { text: "Flujo", link: "/es/leaf-detection/segmentation-current-flow" },
          { text: "Entrenamiento", link: "/es/leaf-detection/segmentation-cloud-training" },
          { text: "Mejoras", link: "/es/leaf-detection/segmentation-improvement-protocol" },
          { text: "Resultados", link: "/es/leaf-detection/segmentation-d01-results" },
          { text: "Decisiones", link: "/es/decisions/adr-leaf-instance-segmentation-strategy" },
        ],
        sidebar: { "/es/": segmentationSidebar },
        search: { provider: "local" },
        outlineTitle: "En esta página",
        docFooter: { prev: "Anterior", next: "Siguiente" },
        logo: "/logo.svg",
      },
    },
  },
  themeConfig: { search: { provider: "local" } },
});
