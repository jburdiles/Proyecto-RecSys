# RecSys Restaurantes - Recomendación Multimodal con Yelp

Proyecto final de Sistemas de Recomendación - Joaquín Burdiles, Alonso Tamayo, Marcelo Vargas.  
Pontificia Universidad Católica de Chile · IIC3633 · 2026-1

Exploramos recomendación de restaurantes combinando **filtrado colaborativo**, **texto** (reseñas, SBERT) e **imágenes** (fotos, CLIP) del dataset público de Yelp, incluyendo modelos basados en redes de grafos multimodales con fusión por Dempster-Shafer.

---

## Problema

Recomendar restaurantes a usuarios basándose en su historial de visitas, incorporando modalidades visuales y textuales. El objetivo es evaluar si las imágenes y el texto de los locales aportan información complementaria al filtrado colaborativo puro.

---

## Dataset

Dataset público de Yelp: https://business.yelp.com/data/resources/open-dataset/

Archivos necesarios en `data/raw/`:

| Archivo | Descripción |
| --- | --- |
| `yelp_academic_dataset_business.json` | Negocios - filtrado a restaurantes |
| `yelp_academic_dataset_review.json` | Reviews de usuarios |
| `photos.json` | Metadata de fotos (label: food / inside / outside / drink / menu) |
| `photos/*.jpg` | Imágenes de los restaurantes |

Los archivos **no se incluyen en el repositorio**. Descargar desde el sitio de Yelp y extraer con:

```bash
tar -xf yelp_dataset.tar -C data/raw/
tar -xf yelp_photos.tar  -C data/raw/
```

**Preprocesamiento** (`02_preprocessing.ipynb`): se parte del dataset completo de Yelp y se aplican, en orden:

1. `build_dataset(min_reviews_per_business=20, min_photos_per_business=5)` — negocios con al menos 20 reviews y 5 fotos.
2. Filtro geográfico a las **top-5 ciudades** con más restaurantes (Philadelphia, Tampa, New Orleans, Nashville, Indianapolis). Configurable con `ONLY_PHILADELPHIA` para restringir a una sola ciudad.
3. Filtro de texto: reviews con al menos 10 palabras tras limpieza.
4. Filtro de actividad: usuarios con al menos 5 reviews (`5-core` por usuario).

**Estadísticas del dataset procesado (top-5 ciudades):**

| Métrica | Valor |
| --- | --- |
| Dataset completo Yelp | 150,346 negocios · 6,990,280 reviews |
| Restaurantes con imagen (≥5 fotos) | 3,824 |
| Fotos | 60,199 |
| Usuarios | 18,117 |
| Reviews | 188,822 |
| Restaurantes efectivamente reseñados | 1,832 |
| Reviews/usuario (promedio) | 10.4 |
| Sparsity de la matriz (18,117 × 1,832) | 99.43% (densidad 0.57%) |

Split temporal 80/20: **156,121 reviews de entrenamiento** · **32,701 de test**

---

## Instalación

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter lab
```

Los modelos de Sentence-Transformers (`all-MiniLM-L6-v2`) y CLIP (`ViT-B/32`) se descargan automáticamente en el primer uso vía HuggingFace.

---

## Estructura del proyecto

```
├── data/
│   ├── raw/               # JSONs + fotos del dataset (no versionados)
│   ├── processed/         # CSVs filtrados y limpios
│   └── embeddings/        # embeddings SBERT y CLIP (.npz, no versionados)
├── notebooks/
│   ├── 01_EDA.ipynb               # exploración del dataset
│   ├── 02_preprocessing.ipynb     # filtrado 5-core y limpieza
│   ├── 03_text_model.ipynb        # generación de embeddings SBERT
│   ├── 04_image_model.ipynb       # generación de embeddings CLIP
│   ├── 05_random.ipynb            # modelo: Random (baseline)
│   ├── 06_mostpopular.ipynb       # modelo: MostPopular (baseline)
│   ├── 07_svd.ipynb               # modelo: SVD (CF puro)
│   ├── 08_textcf.ipynb            # modelo: TextCF (SVD + SBERT)
│   ├── 09_visualcf.ipynb          # modelo: VisualCF (SVD + CLIP)
│   ├── 10_mmgcn.ipynb             # modelo: MMGCN (GCN multimodal, fusión promedio)
│   ├── 11_mmgcn_tmc.ipynb         # modelo: MMGCN-TMC (GCN + Dempster-Shafer)
│   └── 12_comparison.ipynb        # tabla y figura comparativa final
├── src/
│   ├── data_loader.py     # carga línea a línea de los JSONs de Yelp
│   ├── text_features.py   # TextFeatureExtractor (SBERT)
│   ├── image_features.py  # CLIPFeatureExtractor
│   └── evaluation.py      # split temporal + métricas de ranking y rating
├── results/
│   ├── {modelo}/
│   │   ├── metrics.csv    # métricas por K para cada modelo
│   │   └── config.json    # hiperparámetros del modelo
│   ├── eda/               # figuras del análisis exploratorio
│   ├── summary_k10.csv    # tabla resumen a K=10
│   └── comparison.png     # figura comparativa de todas las curvas
└── requirements.txt
```

Cada notebook de modelo (05–11) define su clase directamente y guarda métricas en `results/{modelo}/metrics.csv`. El notebook 12 carga todos esos archivos para la comparación final sin re-entrenar nada.

---

## Cómo reproducir los experimentos

**Paso 1 - Preparación de datos** (correr una vez, en orden):

```
01_EDA.ipynb            -> exploración inicial + figuras en results/eda/
02_preprocessing.ipynb  -> genera data/processed/*.csv + figuras en results/eda/
03_text_model.ipynb     -> genera data/embeddings/text_embeddings.npz
04_image_model.ipynb    -> genera data/embeddings/clip_embeddings.npz
```

**Paso 2 - Entrenar modelos** (independientes entre sí):

```
05_random.ipynb         -> results/random/
06_mostpopular.ipynb    -> results/mostpopular/
07_svd.ipynb            -> results/svd/
08_textcf.ipynb         -> results/textcf/
09_visualcf.ipynb       -> results/visualcf/
10_mmgcn.ipynb          -> results/mmgcn/
11_mmgcn_tmc.ipynb      -> results/mmgcn_tmc/
```

**Paso 3 - Comparación final:**

```
12_comparison.ipynb     -> results/comparison.png + results/summary_k10.csv
```

---

## Modelos implementados

| Modelo | Tipo | Descripción |
| --- | --- | --- |
| **Random** | Baseline | Recomienda ítems no vistos aleatoriamente |
| **MostPopular** | Baseline | Recomienda los restaurantes con más reviews |
| **SVD** | CF puro | TruncatedSVD con 50 factores latentes (scikit-learn) |
| **TextCF** | CF + texto | Combina SVD con similitud coseno sobre embeddings SBERT |
| **VisualCF** | CF + imagen | Combina SVD con similitud coseno sobre embeddings CLIP |
| **MMGCN** | GCN multimodal | GCN bipartito 3 capas, fusión por promedio (texto + imagen), loss BPR+L2 |
| **MMGCN-TMC** | GCN + incertidumbre | Extiende MMGCN con fusión Dempster-Shafer (TMC, Han et al. TPAMI 2023) |

### Detalles de modelos avanzados

**VisualCF:** El perfil visual de cada usuario se construye promediando los embeddings CLIP de los restaurantes que valoró positivamente (rating ≥ 4). El score final mezcla la puntuación SVD normalizada con la similitud coseno al perfil visual: `score = α·svd_norm + (1−α)·cosine_sim` (α\*=0.5).

**MMGCN:** GCN bipartito usuario-ítem (L=3 capas, dim=64) sobre 18,117 usuarios y 1,830 restaurantes multimodales (los que tienen ambas modalidades). Cada usuario tiene un embedding de preferencia aprendible (init xavier); los ítems parten de sus features modales proyectadas por un MLP (imagen CLIP 512d→256, texto SBERT 384d directo) con residual id-embedding en cada capa. Fusión: promedio aritmético de las representaciones GCN de imagen y texto. Loss: BPR + regularización L2. Entrenado con lr=1e-4, batch_size=1024, 50 épocas.

**MMGCN-TMC:** Misma arquitectura GCN que MMGCN. Reemplaza la fusión por promedio con la combinación de Dempster-Shafer del marco TMC (Han et al., TPAMI 2023, DOI: 10.1109/TPAMI.2022.3171983), con una cabeza de evidencia (evidence_dim=16) por modalidad. Loss: BPR + KL-DS con annealing lineal (annealing_step=10) + L2, 150 épocas.

---

## Resultados principales

Evaluación sobre conjunto de test. Split temporal 80/20 (respetando el orden cronológico de reviews por usuario). Métricas de precisión (Precision/Recall/NDCG) y *beyond-accuracy* (Novelty, Coverage, ILD) a K=10, ordenadas por NDCG@10.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Novelty@10 | Coverage@10 | ILD@10 | Δ NDCG vs SVD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **MMGCN** | **0.0198** | **0.1219** | **0.0692** | 6.29 | 0.669 | 0.497 | **+14.4%** |
| MMGCN-TMC | 0.0192 | 0.1200 | 0.0681 | 6.04 | 0.538 | 0.499 | +12.6% |
| VisualCF | 0.0188 | 0.1104 | 0.0629 | 6.07 | 0.318 | 0.504 | +4.0% |
| SVD | 0.0180 | 0.1059 | 0.0605 | 6.09 | 0.245 | 0.516 | - |
| TextCF | 0.0160 | 0.0905 | 0.0528 | 6.52 | 0.667 | 0.402 | −12.7% |
| MostPopular | 0.0071 | 0.0463 | 0.0265 | 4.57 | 0.011 | 0.483 | −56.2% |
| Random | 0.0009 | 0.0049 | 0.0026 | 8.78 | 0.999 | 0.548 | −95.7% |

- **Novelty@10:** self-information media de la lista (mayor = ítems menos populares).
- **Coverage@10:** fracción del catálogo evaluable que el modelo llega a recomendar.
- **ILD@10:** diversidad intra-lista (disimilitud coseno media entre ítems, sobre el espacio de features CLIP).

---

## Notas de implementación

- **Foco geográfico**: el dataset procesado se restringe a las **top-5 ciudades** con más restaurantes (configurable en `02_preprocessing.ipynb`; `ONLY_PHILADELPHIA=True` lo reduce a una sola ciudad).
- Split train/test **temporal**: las últimas reviews de cada usuario van a test, no aleatorio (`temporal_train_test_split`, `src/evaluation.py`).
- Filtrado por actividad: mínimo 5 reviews por usuario y mínimo 20 reviews + 5 fotos por restaurante.
- **Métricas** (`src/evaluation.py`): ranking — Precision@K, Recall@K, NDCG@K, Novelty@K, Hit-Rate@K, Coverage@K e ILD@K; predicción de rating — RMSE y MAE (`evaluate_rmse`).
- Los embeddings se guardan en `data/embeddings/` y se cargan sin recalcular en ejecuciones posteriores.
- **Cobertura multimodal**: los embeddings CLIP cubren los 3,824 restaurantes (VisualCF), mientras que los embeddings de texto SBERT cubren los 1,830 restaurantes reseñados en train. Los modelos GCN (MMGCN / MMGCN-TMC) operan sobre esos 1,830 restaurantes con ambas modalidades.
- Todo corre **localmente**, sin APIs de pago ni modelos propietarios.

---

## Disclaimer

Parte de la redacción de este README y de la documentación del proyecto fue asistida por herramientas de inteligencia artificial (IA generativa) con el fin de acelerar el proceso de escritura y estructuración. Asimismo, la IA fue utilizada como apoyo para adaptar arquitecturas y modelos existentes de la literatura (como MMGCN y Dempster-Shafer TMC) a las particularidades de este proyecto y dataset. Todo el contenido técnico, los experimentos, los resultados y las conclusiones son propios del equipo y fueron verificados manualmente.
