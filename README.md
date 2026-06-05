# RecSys Restaurantes - Recomendación Multimodal con Yelp

Proyecto final de Sistemas de Recomendación - Joaquín Burdiles, Marcelo Vargas, Alonso Tamayo.  
Pontificia Universidad Católica de Chile · IIC3633 · 2026-1

Exploramos recomendación de restaurantes combinando **filtrado colaborativo**, **texto** (reviews, SBERT) e **imágenes** (fotos, CLIP) del dataset público de Yelp, incluyendo modelos basados en redes de grafos multimodales con fusión por Dempster-Shafer.

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

**Estadísticas del dataset procesado:**

| Etapa | Usuarios | Restaurantes | Reviews |
| --- | --- | --- | --- |
| Dataset completo Yelp | - | 150,346 negocios | 6,990,280 |
| Filtrado a restaurantes | - | 52,286 | - |
| Filtrado 5-core + fotos | 10,490 | 1,151 | 100,447 |

Split temporal 80/20: **83,256 reviews de entrenamiento** · **17,191 de test**

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
│   ├── 11_mmgcn_edl.ipynb         # modelo: MMGCN-TMC (GCN + Dempster-Shafer)
│   └── 12_comparison.ipynb        # tabla y figura comparativa final
├── src/
│   ├── data_loader.py     # carga línea a línea de los JSONs de Yelp
│   ├── text_features.py   # TextFeatureExtractor (SBERT)
│   ├── image_features.py  # CLIPFeatureExtractor
│   └── evaluation.py      # Precision@K, Recall@K, NDCG@K, split temporal
├── results/
│   ├── {modelo}/
│   │   ├── metrics.csv    # métricas por K para cada modelo
│   │   └── config.json    # hiperparámetros del modelo
│   ├── eda/               # figuras del análisis exploratorio
│   ├── summary_k10.csv    # tabla resumen a K=10
│   └── comparison.png     # figura comparativa de todas las curvas
├── docs/
│   ├── H1_Propuesta.tex   # propuesta del proyecto (hito 1)
│   └── h2_informe.tex     # informe intermedio (hito 2)
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
11_mmgcn_edl.ipynb      -> results/mmgcn_tmc/
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

**MMGCN:** GCN bipartito usuario-ítem (LightGCN, L=3 capas, dim=64). Usuarios inicializados con factores SVD proyectados; ítems con features modales proyectadas (dim_latent_v=256 para imagen, 384d directo para texto). Fusión: promedio aritmético de representaciones GCN. Loss: BPR + L2. Entrenado con lr=1e-4, batch_size=1024, 50 épocas.

**MMGCN-TMC:** Misma arquitectura GCN que MMGCN. Reemplaza la fusión por promedio con la combinación de Dempster-Shafer del marco TMC (Han et al., TPAMI 2023, DOI: 10.1109/TPAMI.2022.3171983). Loss: BPR + KL-DS con annealing lineal + L2.

---

## Resultados principales

Evaluación sobre conjunto de test. Split temporal 80/20 (respetando el orden cronológico de reviews por usuario).

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Δ vs SVD |
| --- | --- | --- | --- | --- |
| **MMGCN** | **0.0233** | **0.1538** | **0.0883** | **+17.9%** |
| MMGCN-TMC | 0.0222 | 0.1458 | 0.0829 | +10.7% |
| VisualCF | 0.0210 | 0.1351 | 0.0761 | +1.6% |
| SVD | 0.0204 | 0.1326 | 0.0749 | - |
| TextCF | 0.0170 | 0.1049 | 0.0619 | −17.4% |
| MostPopular | 0.0077 | 0.0506 | 0.0287 | −61.7% |
| Random | 0.0012 | 0.0077 | 0.0038 | −94.9% |

---

## Notas de implementación

- Split train/test **temporal**: las últimas reviews de cada usuario van a test, no aleatorio.
- Filtrado **5-core**: mínimo 5 interacciones por usuario y por restaurante.
- Los embeddings se guardan en `data/embeddings/` y se cargan sin recalcular en ejecuciones posteriores.
- **Cobertura multimodal**: los modelos GCN operan sobre los 1,151 restaurantes con ambas modalidades; VisualCF puede recomendar de los 3,824 con imagen.
- Todo corre **localmente**, sin APIs de pago ni modelos propietarios.

---

## Disclaimer

Parte de la redacción de este README y de la documentación del proyecto fue asistida por herramientas de inteligencia artificial (IA generativa) con el fin de acelerar el proceso de escritura y estructuración. Asimismo, la IA fue utilizada como apoyo para adaptar arquitecturas y modelos existentes de la literatura (como MMGCN y Dempster-Shafer TMC) a las particularidades de este proyecto y dataset. Todo el contenido técnico, los experimentos, los resultados y las conclusiones son propios del equipo y fueron verificados manualmente.
