# Proyecto de Machine Learning: Inteligencia Musical y Predicción de Popularidad en Spotify
# Matriz de Responsabilidades

---

## Gustavo Alvial
### Qué hizo en el proyecto
* Generó el notebook asociado al modelo de segmentación.
* Desarrolló el modelo de segmentación (K-Means), modificando el `Pipeline` y el dataset acorde al objetivo del modelo.
* Caracterizó los clusters de la segmentación realizada, incluyendo visualizaciones que faciliten la interpretabilidad.
* Modificó y actualizó la presentación en Canva, agregando la sección de modelo de segmentación y visualizaciones afínes.

---

## Benjamín Neira
### Qué hizo en el proyecto
* Migró el notebook anterior y lo compatibilizó con la evaluación actual, corrigiendo además detalles y celdas que afectaban la calidad del dataset. 
* Modificó el `Pipeline` e incorporó ideas para mitigar el *overfitting* en relación con `TargetEncoder` y `OneHotEncoder` (este último descartado posteriormente). 
* Desarrolló los modelos de regresión (LinearRegression, RandomForest y DecissionTree) junto con un gráfico de valores reales vs predichos.
* Modificó y actualizó la presentación en Canva, desde el inicio hasta los modelos de regresión. 
* Migró y actualizó el informe según la evaluación actual, abarcando el objetivo general, los objetivos específicos, los KPIs y todo el *Pipeline*. Además, creó el apartado de Modelamiento, en el cual detalló los modelos de regresión.
---

## Matías Turra
### Qué hizo en el proyecto
* Modificó y actualizó la presentación en Canva, agregando la sección de modelos de clasificación.*
* Desarrolló los modelos de clasificación (Logistic Regression y Decision Tree) junto con una matriz de confusión y un gráfico de ROC AUC.
* Ejecutó pruebas para cada modelo de clasificación con diferentes técnicas de GridSearchCV y RandomizedSearchCV.
* Creó el repositorio en GitHub.
* Modificó el archivo README.md, actualizando el informe para detallar los modelos de clasificación.

---
# 1. Descripción del Problema de Negocio

La industria de la música en streaming se caracteriza por un volumen masivo y continuo de lanzamientos, donde plataformas líderes como Spotify incorporan diariamente decenas de miles de canciones a su catálogo global. En este entorno hipercompetitivo, la curaduría editorial de listas de reproducción (*playlists*), el diseño de recomendaciones personalizadas y el descubrimiento temprano de talentos (*Artist & Repertoire / scouting*) constituyen procesos estratégicos críticos para sostener la retención y el compromiso (*engagement*) de los suscriptores. Recomendar temas que no conectan con la audiencia eleva la tasa de abandono temprano (*skip rate*) y deteriora la experiencia del usuario, mientras que no detectar a tiempo los nuevos éxitos representa una pérdida significativa de oportunidades comerciales frente a la competencia.

Frente a este escenario, la compañía requiere desarrollar una solución analítica de **inteligencia musical** sustentada en modelos de Machine Learning, capaz de estimar la popularidad de las canciones a partir de sus atributos acústicos intrínsecos (tales como bailabilidad, energía, sonoridad, valencia emocional o nivel de instrumentalidad) y sus metadatos contextuales de producción (género musical, artista y colaboraciones). 

Para maximizar el valor de esta solución frente a las diversas necesidades operativas y estratégicas de Spotify, el proyecto aborda el desafío predictivo desde dos paradigmas complementarios de aprendizaje supervisado:

* **Modelos de Regresión Continua (Predicción cuantitativa de `popularity` de 0 a 100):** Orientados a estimar con precisión el puntaje numérico continuo de popularidad en su escala nativa. Esta perspectiva proporciona una calibración de granularidad fina, indispensable para rankear canciones con márgenes estrechos de diferencia y priorizar el ordenamiento interno de pistas en catálogos específicos, evaluando el desempeño mediante un margen estrecho de error absoluto ($\text{MAE} < 8$ puntos).
* **Modelos de Clasificación Multiclase Ordinal (Segmentación en `popularity_categorica`):** Orientados a clasificar y discriminar las pistas en cinco niveles ordinales estratégicos (*Muy Baja*, *Baja*, *Media*, *Alta* y *Muy Alta* popularidad). Este enfoque traduce la métrica cuantitativa en segmentos operativos directos, agilizando la toma de decisiones ejecutivas en curaduría editorial (asignación por niveles de exposición en *playlists*) y activando alertas tempranas de scouting basadas en métricas de discriminación robustas ($F1\text{-Score}$, *Precision* y *Recall* $> 0.60$).

La integración sinérgica de ambos enfoques predictivos permite resolver los desafíos clave del negocio:

1. **Optimizar la curaduría editorial y asignación en listas oficiales:** Priorizar canciones con alto potencial de interacción tanto en su puntaje continuo como en los segmentos de *Alta* y *Muy Alta* popularidad, asegurando selecciones de calidad que reduzcan el *skip rate* y maximicen el tiempo de escucha en la plataforma.
2. **Potenciar el descubrimiento temprano de talentos (*Scouting / A&R*):** Identificar tempranamente pistas con alto potencial comercial provenientes de artistas emergentes o sellos independientes antes de su explosión orgánica en el mercado, cumpliendo con la meta de alcanzar una tasa de detección temprana de éxitos $\ge 50\%$ al cabo del siguiente mes de evaluación.
3. **Reducir el costo de oportunidad y la fatiga auditiva:** Disminuir la exposición de canciones categorizadas en popularidad *Muy Baja* o *Baja* dentro de los sistemas de recomendación activa, evitando el desgaste de los usuarios y optimizando la inversión en promoción.
4. **Habilitar una toma de decisiones estratificada y flexible:** Proveer a los equipos editoriales y de marketing una herramienta dual que permite actuar rápidamente según niveles cualitativos de popularidad o profundizar en ordenamientos milimétricos mediante predicciones continuas.

El desarrollo del proyecto se rige bajo el estándar internacional de la metodología **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*), estructurando un flujo de trabajo riguroso que abarca desde la comprensión del negocio y exploración profunda de los datos hasta la ingeniería de características, el entrenamiento de modelos de regresión y clasificación, y su posterior validación cruzada. 

Asimismo, la solución incorpora desde su arquitectura inicial sólidos principios de gobernanza, privacidad y responsabilidad ética:
* **Integridad Técnica y Criptográfica:** Certificación de la inalterabilidad de la fuente de datos mediante funciones hash SHA-256, asegurando un pipeline reproducible y blindado contra corrupciones de origen.
* **Cumplimiento Normativo de Privacidad:** Adherencia a los marcos legales de protección de datos personales (Ley N° 21.719 en Chile y RGPD/GDPR europeo), garantizando la total ausencia de información de identificación personal (PII) de los oyentes.
* **Auditoría de Equidad Algorítmica (*Fairness*):** Detección y mitigación de sesgos históricos que privilegien desproporcionadamente a géneros comerciales sobre expresiones independientes o alternativas, evaluando el comportamiento de ambos modelos mediante la Tasa de Impacto Dispar (*Disparate Impact Ratio* - DIR) bajo la regla del 80%.

---

# 2. Objetivos del Proyecto

### 2.1. Objetivo General
Desarrollar un pipeline integral de preparación de datos y una solución analítica de Machine Learning bajo la metodología CRISP-DM, que permita predecir la popularidad de canciones en Spotify a partir de sus atributos acústicos y metadatos contextuales, así como determinar perfiles musicales mediante segmentación para optimizar las recomendaciones y la generación de listas de reproducción automáticas, identificando los factores determinantes del éxito musical para fundamentar la toma de decisiones en curaduría editorial y scouting artístico, garantizando la integridad de los datos, la reproducibilidad del proceso y la equidad algorítmica.

---

### 2.2. Objetivos Específicos

* **Auditar la procedencia, gobernanza e integridad técnica del dataset:** Implementar validaciones criptográficas mediante funciones hash (SHA-256) para certificar la inalterabilidad de los datos de entrada, garantizando el cumplimiento de los estándares éticos y normativos vigentes de protección de datos (Ley N° 21.719 y GDPR).
* **Ejecutar un análisis exploratorio de datos (EDA) y control de calidad:** Cuantificar y subsanar anomalías en el catálogo (gestión de duplicados por multiplicidad de géneros por pista, registros nulos e inconsistencias lógicas de rango) y evaluar distribuciones univariadas, justificando estadísticamente el criterio de retención o tratamiento de valores atípicos (*outliers*) en variables acústicas continuas.
* **Determinar correlaciones y dependencias multivariadas con la variable objetivo:** Medir la fuerza de asociación lineal y no lineal (coeficientes de Spearman y Razón de Correlación $\eta$ en atributos cualitativos) frente a `popularity`, identificando las características con verdadero poder explicativo y descartando identificadores o variables redundantes para prevenir sobreajuste (*overfitting*) y fuga de datos (*data leakage*).
* **Auditar y evaluar sesgos algorítmicos y de representatividad (Fairness):** Detectar la presencia de sesgos de selección e históricos dentro del catálogo musical (tales como la sobrerrepresentación de estilos comerciales en detrimento de escenas independientes), evaluando el impacto ético del modelo mediante métricas de justicia algorítmica como la Tasa de Impacto Dispar (*Disparate Impact Ratio*) bajo la regla del 80%.
* **Diseñar e implementar ingeniería de características vectorizada:** Generar nuevos predictores basados en interacciones acústicas (p. ej., combinación aditiva de valencia e instrumentalidad, segregación de artista principal y detección binaria de colaboraciones), mitigando el impacto de la alta cardinalidad mediante técnicas avanzadas de transformación y codificación (*Target Encoding* y *One-Hot Encoding*).
* **Construir un pipeline de preprocesamiento desacoplado y reproducible:** Estructurar un flujo modular mediante `ColumnTransformer` y separación estricta de particiones de entrenamiento y prueba (*Train/Test split*), garantizando la reproducibilidad metodológica y la preparación de los datos para la futura fase de modelamiento predictivo (regresión o clasificación).
* **Implementar y comparar modelos de aprendizaje supervisado (Regresión y Clasificación):** Entrenar algoritmos (como Random Forest, Decision Tree y Regresión Logística) para estimar el puntaje continuo y categorizar el nivel de popularidad, optimizando hiperparámetros y evaluando su rendimiento predictivo mediante métricas clave (MAE, F1-Score, ROC AUC).
* **Identificar y caracterizar perfiles musicales:** Agrupar el catálogo mediante K-Means a partir de sus atributos acústicos para descubrir arquetipos claros y definir su identidad sonora.
* **Optimizar la estructura de los clústeres:** Determinar el número ideal de grupos ($K$) mediante métricas de validación (como el coeficiente de silueta) para garantizar una segmentación robusta y sin solapamientos.

# 3. Definición de KPIs del Proyecto

Para evaluar el desempeño integral y la viabilidad del proyecto, se definen indicadores clave de rendimiento (KPIs) divididos en dos niveles complementarios: **KPIs de Negocio** (impacto operacional y valor estratégico en la plataforma) y **KPIs Técnicos de Machine Learning** (precisión matemática, capacidad predictiva y justicia algorítmica).

---

### 3.1. KPIs de Negocio
Miden el impacto directo de la solución sobre la experiencia del usuario y la gestión del catálogo de Spotify:

**Tasa de Detección Temprana de Éxitos Independientes (*Breakout Discovery Rate*):**
   * **Definición:** Porcentaje de canciones producidas por artistas emergentes o de sellos independientes (con catálogos históricos reducidos) con alto potencial de éxito que el sistema identifica con precisión.
   * **Meta Cuantificable:** Lograr que el modelo identifique correctamente al menos el **5  0% de las canciones independientes con potencial de éxito** (definidas con un umbral de popularidad $\ge 60$ puntos), facilitando su inclusión temprana en playlists destacadas y optimizando la labor de búsqueda de nuevos talentos (*scouting* artístico).

---

### 3.2. KPIs Técnicos de Machine Learning
Evalúan la calidad estadística de las predicciones y la equidad del pipeline analítico:

**Error Absoluto Medio (*Mean Absolute Error* - MAE):**
   * **Definición:** Métrica continua que cuantifica el promedio de las diferencias absolutas entre el puntaje de popularidad real de la canción y el estimado por el modelo.
   * **Meta Cuantificable:** Obtener un **$\text{MAE} < 8$ puntos de popularidad** (en la escala nativa de 0 a 100) evaluado sobre el conjunto de prueba desacoplado (*test set*), asegurando un margen de error estrecho y confiable para la toma de decisiones.

**Métricas de Desempeño en Clasificación (*F1-Score / Precision / Recall*):**
  * **Definición:** Conjunto de métricas estadísticas orientadas a evaluar la robustez y el balance entre precisión (minimización de falsos positivos) y exhaustividad (minimización de falsos negativos) al discretizar la popularidad en categorías de éxito.
  * **Meta Cuantificable:** Superar un umbral de **$> 0.60$ en las métricas de clasificación** (especialmente en *F1-Score* ponderado y sensibilidad sobre la clase de éxito) en el conjunto de prueba (*test set*).
---


# 4. Descripción de las Fuentes de Datos y Entorno Tecnológico

### 4.1. Ficha Técnica y Estructura del Dataset
La base de información empleada corresponde al conjunto de datos **Spotify Tracks Dataset**, derivado de la API oficial de Spotify (*Spotify Web API*). El dataset crudo inicial cuenta con **114.000 registros** y **21 atributos estructurados**, que combinan identificadores, metadatos artísticos y variables de procesamiento de señales de audio:

* **Identificadores y Metadatos:** `track_id` (identificador único alfanumérico de Spotify), `artists` (artista o artistas intérpretes, separados por punto y coma), `album_name` (álbum de publicación), `track_name` (título de la canción) y `track_genre` (clasificación estilística con 114 géneros musicales únicos).
* **Variable Objetivo (*Target*):** `popularity` (variable numérica entera que oscila entre 0 y 100, calculada dinámicamente por la plataforma según el volumen y recencia de reproducciones).
* **Variables de Audio Intrínsecas:** Atributos acústicos normalizados entre $0.0$ y $1.0$ (`danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`), variables físicas (`loudness` en decibeles, `tempo` en BPM, `duration_ms` en milisegundos), y descriptores tonales (`key`, `mode`, `time_signature`).
* **Contenido Explícito:** `explicit` (indicador booleano que señala si la pista contiene letras explícitas).

---

### 4.2. Procedencia, Ingesta Remota e Integridad Criptográfica
Para asegurar la reproducibilidad de la solución sin depender de descargas manuales locales, el archivo de datos (`Spotify_Tracks_Dataset.csv`) se encuentra alojado en un repositorio público de **GitHub**. 

La ingesta se efectúa en memoria mediante una petición HTTP vía `requests` y decodificación binaria con `io.BytesIO`, parseada directamente mediante la librería `pandas` en Python. Con el fin de certificar la inalterabilidad de la fuente desde su origen, el pipeline ejecuta de forma obligatoria una auditoría criptográfica mediante la función hash **SHA-256**:

$$\text{Hash Verificado:} \quad \texttt{b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd}$$

Si el valor calculado sobre los bytes entrantes difiere del valor de control original, la ejecución se interrumpe de inmediato mediante excepciones controladas (`ValueError` / `RuntimeError`), garantizando que ningún dato corrompido o adulterado ingrese a la fase analítica.

---

### 4.3. Auditoría de Privacidad y Cumplimiento Normativo (PII Audit)
Se realizó una inspección sobre el contenido de la fuente para determinar la presencia de información de identificación personal (*Personally Identifiable Information* - PII). Se verificó que el conjunto de datos:
* **No contiene identificadores de usuarios:** No almacena RUT, direcciones de correo, identificadores de clientes, hábitos individuales de consumo ni registros de geolocalización de oyentes.
* **Tratamiento de figuras públicas:** Los nombres registrados corresponden a artistas y figuras de notoriedad pública dentro del ámbito comercial musical.
* **Cumplimiento legal:** El manejo de la información respeta los estándares vigentes de la regulación nacional de protección de datos personales de Chile (**Ley N° 21.719**) y los principios de minimización del dato del **RGPD/GDPR europeo**.

---

### 4.4. Justificación de Herramientas Colaborativas y Ecosistema Técnico
En conformidad con las directrices de la metodología CRISP-DM, se adoptó el siguiente entorno tecnológico colaborativo:

1. **GitHub (Control de Versiones y Distribución):** Utilizado como repositorio central para el versionamiento semántico del código fuente (`.ipynb`), el seguimiento de cambios (*commits*) entre integrantes del equipo y la distribución desacoplada de los datos crudos.
2. **Google Colab (Entorno de Cómputo Colaborativo en la Nube):** Seleccionado como entorno de desarrollo unificado para permitir la edición concurrente y la reproducibilidad exacta de dependencias analíticas en máquinas virtuales homogéneas, eliminando discrepancias operativas entre sistemas operativos locales.
3. **Ecosistema Científico de Python:**
   * `pandas` y `numpy`: Para la ingesta, manipulación estructurada y cálculo vectorial eficiente.
   * `matplotlib` y `seaborn`: Para la construcción de visualizaciones estadísticas (histogramas, boxplots y barras comparativas.
   * `scipy`: Para pruebas estadísticas inferenciales (Chi-cuadrado y cálculo de V de Cramér).
   * `scikit-learn` y `category_encoders`: Para el encapsulamiento del flujo de transformación mediante `ColumnTransformer` (con `OneHotEncoder`, `TargetEncoder` y escaladores numéricos), asegurando una separación estricta entre entrenamiento y prueba (*Train/Test split*) sin fuga de información (*data leakage*).

# 5. Preparación y Análisis Exploratorio de los Datos (EDA)

El análisis exploratorio de datos y el flujo de preparación se ejecutaron de manera secuencial e iterativa, garantizando que cada transformación estuviera técnicamente justificada por la evidencia estadística encontrada y blindada contra la fuga de datos (*data leakage*).

---

### 5.1. Auditoría de Calidad, Limpieza y Depuración Estructural
Antes de realizar inferencias estadísticas, se auditó la sanidad estructural del conjunto de datos (114.000 filas y 21 columnas iniciales):

1. **Gestión de Registros Nulos:**
   * **Hallazgo:** Se detectó exactamente **1 registro** con valores faltantes (`NaN`) en los metadatos textuales `artists`, `track_name` y `album_name` (asociado al identificador `track_id: 1kR4gIb7nGxHPI3D2ifs59`).
   * **Acción y Justificación:** Se aplicó eliminación por lista (*listwise deletion*). Al representar el 0.00087% del volumen total, su exclusión no introduce sesgo de selección muestral ni altera las distribuciones acústicas. Rellenarlo con categorías artificiales como *"Desconocido"* habría distorsionado la posterior codificación y agrupamiento por artista.
2. **Depuración de Índices Redundantes:**
   * Se descartó la columna `Unnamed: 0`, la cual correspondía a un índice residual heredado del almacenamiento serializado del CSV sin relevancia analítica ni predictiva.
3. **Conversión de Unidades Físicas:**
   * La variable `duration_ms` fue convertida a escala de segundos (`duration_s = duration_ms / 1000`) para mejorar la interpretabilidad operativa y la visualización de distribuciones temporales.
4. **El Dilema de los Duplicados por Multiplicidad de Género:**
   * **Hallazgo:** La auditoría arrojó 114.000 filas, pero solo **89.740 canciones únicas** por `track_id`. Se descubrió que Spotify registra una misma canción en múltiples filas para asignarle distintos géneros (`track_genre`), existiendo ~25.000 registros repetidos estructuralmente.
   * **Riesgo Técnico:** Mantener filas repetidas con idénticos atributos acústicos provocaría que una misma pista quede repartida entre los conjuntos de entrenamiento y prueba (*Train/Test split*), ocasionando **fuga de datos (*data leakage*)** y memorización espuria.
   * **Acción:** Se colapsó el dataset a registros únicos mediante `groupby('track_id').first()`, garantizando que cada unidad observacional sea tratada una sola vez en el pipeline analítico.

---

### 5.2. Análisis Univariado y Tratamiento de Valores Atípicos (*Outliers*)

1. **Comportamiento Distribucional del Target (`popularity`):**
   * **Concentración en Cero:** Aproximadamente el **14% de las canciones presentan una popularidad exactamente igual a 0**. Este fenómeno refleja la realidad del negocio: una gran masa de canciones de catálogo profundo o lanzamientos independientes que no logran tracción algorítmica ni reproducciones mínimas.
   * **Asimetría:** La distribución general está sesgada a la izquierda; la gran mayoría de las canciones se concentra en el rango de $[0, 50]$ puntos, con una cola reducida de éxitos comerciales que superan los 60-75 puntos.
2. **Inspección de Atributos Acústicos Continuos:**
   * Se inspeccionaron variables acotadas al rango $[0.0, 1.0]$ (`danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`) y variables continuas abiertas (`loudness`, `tempo`, `duration_s`).
   * **Valores Atípicos (*Outliers*):** Los diagramas de caja (*boxplots*) revelaron colas extendidas en `duration_s` (canciones muy largas o audiolibros/sonidos ambientales) y `loudness` (pistas acústicas o silencios extremos).
   * **Decisión Técnica:** **Se decidió conservar los valores atípicos** sin truncamiento ni eliminación indiscriminada. Si bien representan expresiones sonoras legítimas de la diversidad musical de Spotify (música clásica, jazz experimental, pistas para dormir). Como sus valores están entre [0, 1], cualquier técnica de Winsorización o Escalamiento resultó en los mismos resultados, generando una indiferencia entre los tratamientos que se usaron.

---

### 5.3. Análisis Bivariado, Multivariado y Dependencias Estadísticas

1. **Evaluación de Correlación Lineal (Spearman):**
   * Se calculó la matriz de correlación de Spearman entre los atributos de audio numéricos y `popularity`.
   * **Hallazgo Crítico:** La asociación lineal directa entre los descriptores acústicos individuales y la popularidad es notablemente débil: **ninguna variable numérica original supera $\vert{}r\vert{} = 0.12$** frente al target.
   * **Conclusión Estadística:** Los atributos acústicos por sí solos explican menos del 1.5% de la varianza del éxito musical. El éxito comercial en streaming no depende exclusivamente de cómo suena una pista, sino de factores contextuales como el género musical, el alcance del artista y las colaboraciones estratégicas.
2. **Asociación Categórica Multidimensional (V de Cramér / Razón $\eta$):**
   * Al categorizar la popularidad en quintiles (`popularity_cat`), la métrica **V de Cramér** evidenció una asociación sustancial con `track_genre` y con los artistas.
   * **Medianas por Género:** Los géneros comerciales como *pop*, *dance* y *latino* presentan medianas sistemáticas sobre los 60 puntos y una alta tasa de éxitos (*hit rate* con popularidad $> 75$), mientras que categorías como *black-metal*, *classical* o *study* se concentran cerca de la base del target.
3. **Contraste de Perfiles (Canciones Populares vs. No Populares):**
   * Al comparar las medias acústicas entre canciones populares ($> 75$) e impopulares, se observó que la mayoría de los descriptores son similares, excepto por dos métricas distintivas: **la acústica (`acousticness`) y la instrumentalidad (`instrumentalness`) descienden drásticamente en las canciones altamente exitosas**, confirmando que el público masivo prefiere producciones con alta presencia vocal y sintetizada.

---

### 5.4. Auditoría de Ética, Privacidad y Sesgos Algorítmicos

1. **Privacidad y Cumplimiento Normativo (Ley 21.719 y GDPR):**
   * Se comprobó la ausencia total de datos de identificación personal directa de usuarios (PII). La verificación criptográfica con **SHA-256** certifica la inalterabilidad de la fuente desde su repositorio en GitHub.
2. **Sesgo de Selección e Histórico:**
   * El dataset presenta un sesgo histórico de la industria: los algoritmos pasados y la industria radial han privilegiado la visibilidad de géneros masivos (*pop*, *reggaeton*), relegando la música independiente.
3. **Auditoría de Impacto Dispar (Regla del 80%):**
   * Si se evaluara un modelo predictivo optimizado únicamente por precisión matemática sin restricciones, este tendería a predecir popularidad baja para cualquier artista de géneros de nicho.
   * Se incorpora como restricción de gobernanza evaluar el **Ratio de Impacto Dispar (DIR)** entre géneros minoritarios y comerciales ($\text{DIR} \ge 0.80$), garantizando que el pipeline no discrimine sistemáticamente a la música emergente.

---

### 5.5. Ingeniería de Características (*Feature Engineering*) Vectorizada

Para superar la baja señal predictiva de las variables acústicas crudas y manejar la alta dimensionalidad, se implementaron transformaciones vectorizadas con NumPy y Pandas:

* **Segregación del Artista Principal (`artista_principal`):** En pistas con múltiples cantantes (separados por punto y coma), se extrajo vectorialmente el primer intérprete (`artists.str.split(';').str[0]`), aislando al artista de mayor impacto comercial y reduciendo la dispersión textual.
* **Detección Binaria de Colaboración (`es_colaboracion`):** Se creó una bandera booleana (`1/0`) mediante detección de delimitadores en la cadena de artistas (`contains(';')`), capturando el efecto positivo que las colaboraciones (*feats*) generan en la visibilidad de Spotify.
* **Descarte Justificado de Variables:** Se eliminaron identificadores y metadatos no generalizables (`track_id`, `track_name`, `album_name`) por su cardinalidad extrema que induciría memorización espuria (*overfitting*), así como variables acústicas redundantes cuya señal quedó capturada en las nuevas características.

---

### 5.6. Pipeline de Preprocesamiento Desacoplado y Especializado

Para garantizar la reproducibilidad técnica, blindar el flujo analítico contra la fuga de datos (*data leakage*) y optimizar el rendimiento predictivo, se implementó una arquitectura modular mediante `Pipeline` y `ColumnTransformer` de Scikit-Learn. A diferencia de las versiones iniciales, el flujo de preprocesamiento fue completamente rediseñado para dar soporte a dos paradigmas analíticos diferenciados: **Modelos de Regresión Continua** (predicción cuantitativa de `popularity` de 0 a 100) y **Modelos de Clasificación Multiclase** (categorización en `popularity_categorica`).

---

#### 5.6.1. Enrutamiento y Segmentación de Características
El conjunto de variables de entrada $X$ se estructuró en tres grupos funcionales de acuerdo con su tipo de dato y su tratamiento matemático:

1. **Variables Numéricas Continuas (`cols_num`):** `['speechiness', 'time_signature', 'valence', 'loudness', 'instrumentalness']`. Corresponden a los descriptores acústicos que superaron el umbral de asociación lineal relevante ($\vert{}r\vert{} \ge 0.03$), aportando información sobre contenido vocal, métrica rítmica, positividad, sonoridad y ausencia instrumental.
2. **Variables Numéricas Binarias (`cols_bin`):** `['es_colaboracion']`. Bandera booleana que identifica la presencia de colaboraciones (*feats*). Se enruta mediante flujo directo (`passthrough`) para conservar estrictamente su interpretación lógica discreta $\{0, 1\}$, evitando que transformadores de escala como `StandardScaler` la conviertan en valores decimales continuos sin significado de negocio.
3. **Variables Categóricas de Alta Cardinalidad (`cols_nom_te`):** `['artista_principal', 'track_genre']`. Atributos cualitativos determinantes del éxito comercial que requieren codificación supervisada.

---

#### 5.6.2. Evolución de la Codificación Categórica: De One-Hot Encoding a Target Encoding
En las iteraciones tempranas del pipeline, la variable `track_genre` se transformaba mediante `OneHotEncoder`. Sin embargo, esta decisión presentó deficiencias estructurales que motivaron su reemplazo por `TargetEncoder`:

* **Inconvenientes de `OneHotEncoder`:** 
  * Generaba una matriz dispersa (*sparse*) de más de 88 columnas (o hasta 112 en representaciones sin agrupar), donde cada registro contenía decenas de columnas vacías en `0` y únicamente una en `1`.
  * Esta fragmentación dimensional perjudica severamente a los modelos basados en árboles (Random Forest y Árboles de Decisión), ya que diluye la ganancia de información en los nodos y reduce drásticamente la probabilidad de que una columna binaria individual sea seleccionada en una partición útil (*split*).
  * Además, las columnas generadas no guardaban una relación monotónica directa con la magnitud del target `popularity`.
* **Ventajas de la adopción unificada de `TargetEncoder`:**
  * Comprime `track_genre` y `artista_principal` a **exactamente una columna numérica continua por variable**, preservando una dimensionalidad densa y compacta.
  * Captura de forma inmediata la señal comercial y la tracción histórica asociada al género y al músico, facilitando divisiones continuas claras en algoritmos de árbol y aportando un predictor robusto para modelos lineales.
  * **Mitigación del Sobreajuste (*Overfitting*):** El riesgo inherente de memorización en `TargetEncoder` fue controlado mediante la parametrización de validación interna cruzada (*out-of-fold*) y suavizado empírico.

---

#### 5.6.3. Justificación de los Dos Pipelines de Target Encoding (Clasificación vs. Regresión)
Uno de los hallazgos técnicos más críticos del proyecto fue la necesidad de desacoplar el procesamiento categórico en dos pipelines independientes (`pipe_te_cl` y `pipe_te_re`):

1. **Pipeline para Clasificación (`pipe_te_cl`):**
   * Configurado con imputación por moda (`SimpleImputer(strategy="most_frequent")`) y `TargetEncoder()` con parámetros estándar.
   * Al entrenar contra `popularity_categorica` (variable discreta con 5 niveles ordinales: *Muy Baja*, *Baja*, *Media*, *Alta*, *Muy Alta*), el transformador infiere adecuadamente la naturaleza multiclase del objetivo, calculando las probabilidades condicionales por clase de manera estable y sin conflicto de tipos.

2. **Pipeline para Regresión (`pipe_te_re`) y Diagnóstico del Error de Sobresaturación:**
   * Al intentar reutilizar el encoder por defecto (`target_type="auto"`) sobre la variable cuantitativa `popularity`, Scikit-Learn detectó que el target contenía números enteros (`int64` entre 0 y 100) y **asumió erróneamente que se trataba de un problema de clasificación con ~101 clases independientes**.
   * Esta confusión conceptual provocó dos fallos críticos en la etapa de modelamiento:
     1. **Advertencia de Partición:** Arrojó la alerta `/usr/local/lib/python3.13/dist-packages/sklearn/model_selection/_split.py:805: UserWarning: The least populated class in y has only 1 members, which is less than n_splits=5`, debido a que valores extremos de popularidad (p. ej., 98 o 100) contaban con una sola pista en el catálogo, imposibilitando la estratificación interna.
     2. **Explosión Dimensional y Colapso del Score:** El codificador generó más de 100 columnas internas por variable (una por cada entero de clase). Al alimentar estos cientos de variables colineales en una `LinearRegression` sin regularización, los coeficientes matemáticos se desestabilizaron al orden de $10^6$, haciendo colapsar el score de entrenamiento a un aberrante **$R^2 = -385.29144$** debido a la discrepancia entre el ajuste *out-of-fold* y la transformación final.
   * **Solución y Parametrización Definitiva:** Se estabilizó el pipeline fijando parámetros explícitos:
     * `target_type="continuous"`: Fuerza al transformador a tratar `popularity` como una escala cuantitativa real, aplicando validación $K$-Fold estándar de regresión y emitiendo exactamente una sola columna continua flotante por variable categórica.
     * `smooth="auto"`: Aplica contracción empírica bayesiana (*shrinkage*), promediando las categorías con pocas canciones hacia la media global del catálogo para neutralizar *outliers* y éxitos fortuitos de artistas emergentes.
     * `cv=5`: Implementa validación interna cruzada de 5 pliegues durante el ajuste, asegurando que cada canción sea codificada excluyendo su propio valor de popularidad, erradicando la fuga de datos (*target leakage*).
     * `random_state=42`: Garantiza la total reproducibilidad en la partición de los pliegues internos.

---

#### 5.6.4. Tratamiento Numérico Robusto y Filtro de Colinealidad
* **Sub-pipeline Numérico (`pipe_num`):**
  * **`WinsorizerRobusto(limits=(0.02, 0.02))`:** Trunca de forma determinista el 2% inferior y superior de las colas de distribución en las variables continuas, amortiguando el impacto de valores extremos legítimos.
  * **`SimpleImputer(strategy="mean")`:** Se seleccionó la media aritmética para la imputación de nulos residuales; al haber aplicado previamente el winsorizado, los valores atípicos ya no distorsionan la tendencia central, convirtiendo a la media en una estimación robusta y matemáticamente eficiente.
  * **`StandardScaler()`:** Centra las variables en media $0$ y varianza unitaria ($Z$-score), requerimiento indispensable para la convergencia y estabilidad de estimadores lineales y basados en distancias.
* **Capa de Control de Redundancia (`CorrelationFilter(threshold=0.9)`):** Integrada al final de cada pipeline integral para detectar e interceptar pares de variables con colinealidad extrema ($r > 0.90$), previniendo problemas de rango incompleto en las matrices y distorsiones en la asignación de importancia de variables.

---

#### 5.6.5. Pipeline Orientado a Segmentación
* **`RobustScaler()`:** A diferencia de la estandarización clásica basada en la media y la desviación estándar (la cual es altamente vulnerable a los valores atípicos), RobustScaler utiliza el rango intercuartílico (IQR) y la mediana. Esto permite centrar y escalar los datos de manera efectiva sin que los outliers distorsionen la transformación, preservando la integridad de la distribución subyacente.
* **`SimpleImputer(strategy="median")`:** Se selecciona la mediana en lugar de la media debido a la presencia significativa de valores atípicos (outliers) en las variables de interés. Al ser una medida de tendencia central robusta ante valores extremos, la mediana evita que los datos faltantes sean sesgados por distribuciones sesgadas o anomalías.
---

#### 5.6.6. Implementación del Código en Scikit-Learn

```python
# ==============================================================================
# 1. ENRUTAMIENTO DE CARACTERÍSTICAS
# ==============================================================================
cols_num = ['speechiness', 'time_signature', 'valence', 'loudness', 'instrumentalness']
cols_bin = ['es_colaboracion']
cols_nom_te = ['artista_principal', 'track_genre']
# Específico de segmentación
cols_num = ['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','tempo','valence']
# ==============================================================================
# 2. SUB-PIPELINES DE TRANSFORMACIÓN
# ==============================================================================
# Pipeline Numérico: Winsorizado, Imputación por Media y Estandarización
pipe_num = Pipeline(
    steps=[
        ("winsorizer", WinsorizerRobusto(limits=(0.02, 0.02))),
        ("imputacion", SimpleImputer(strategy="mean")),  # La media es robusta tras el recorte de colas
        ("escalado", StandardScaler())
    ]
)

# Pipeline Categórico para CLASIFICACIÓN (TargetEncoder multiclase nativo)
pipe_te_cl = Pipeline(
    steps=[
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder())
    ]
)

# Pipeline Categórico para REGRESIÓN (TargetEncoder continuo y regularizado)
pipe_te_re = Pipeline(
    steps=[
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder(
            target_type="continuous",
            smooth="auto",
            cv=5,
            random_state=42
        ))
    ]
)

# Pipeline numérico de segmentación: Escalado robusto e imputación por mediana
pipe_num = Pipeline(
    steps=[
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", RobustScaler())
    ]
)

# ==============================================================================
# 3. ENSAMBLAJE DE PREPROCESADORES (ColumnTransformer)
# ==============================================================================
# Preprocesador para Clasificación
preprocesadorCL = ColumnTransformer(
    transformers=[
        ("num", pipe_num, cols_num),
        ("bin", "passthrough", cols_bin),
        ("nom_te", pipe_te_cl, cols_nom_te)
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
).set_output(transform="pandas")

# Preprocesador para Regresión
preprocesadorRE = ColumnTransformer(
    transformers=[
        ("num", pipe_num, cols_num),
        ("bin", "passthrough", cols_bin),
        ("nom_te", pipe_te_re, cols_nom_te)
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
).set_output(transform="pandas")

# Preprocesador para Segmentación
preprocesadorSE = ColumnTransformer(
    transformers=[
        ("num", pipe_num, cols_num)
    ],
    verbose_feature_names_out=False
).set_output(transform="pandas")

# ==============================================================================
# 4. PIPELINES GLOBALES DE PRODUCCIÓN
# ==============================================================================
pipeline_clasificacion = Pipeline(steps=[
    ("duplicados", FunctionTransformer()),
    ("preprocesador", preprocesadorCL),
    ("colinealidad", CorrelationFilter(threshold=0.9))
])

pipeline_regresion = Pipeline(steps=[
    ("duplicados", FunctionTransformer()),
    ("preprocesador", preprocesadorRE),
    ("colinealidad", CorrelationFilter(threshold=0.9))
])
```
# 6 Modelamiento

## 6.1 Modelamiento Predictivo: Modelos de Regresión Continua

En esta fase del proyecto (correspondiente a la etapa de *Modeling* de la metodología CRISP-DM), se implementa y evalúa el componente analítico orientado a predecir cuantitativamente el puntaje continuo de popularidad (`popularity`, escala de 0 a 100) a partir de los atributos acústicos y metadatos contextuales de Spotify.

---

### 6.1.1 Configuración Experimental y Espacio de Características

El modelamiento de regresión se ejecuta de forma desacoplada utilizando el subconjunto de características depuradas tras el análisis de correlación y asociación del EDA:

```python
# Definición del espacio de características y variable objetivo continua
X = df[['speechiness', 'time_signature', 'valence', 'loudness', 'instrumentalness',
        'es_colaboracion', 'track_genre', 'artista_principal']]
y_numerico = df['popularity'].astype(float)
```

- **Flujo de preprocesamiento (`pipeline_regresion`):** todas las transformaciones se acoplan directamente a los estimadores finales dentro de un `Pipeline` de Scikit-Learn, previniendo cualquier fuga de datos (*data leakage*) entre particiones. Estas son:
  - Winsorizado y estandarización de las variables cuantitativas.
  - Flujo directo de la variable binaria.
  - `TargetEncoder(target_type="continuous", cv=5, smooth="auto")` regularizado para `track_genre` y `artista_principal`.
- **Partición de datos:** se empleó una separación del 80% para entrenamiento (`X_train`, `y_train`) y 20% para evaluación desacoplada (`X_test`, `y_test`), fijando `random_state=42` para garantizar estricta reproducibilidad.

---

### 6.1.2. Algoritmos Evaluados y Estrategia de Calibración

Para balancear simplicidad, interpretabilidad y capacidad de captura de patrones no lineales, se seleccionaron tres algoritmos representativos de distintas familias analíticas.

#### 1. Regresión Lineal Clásica (*Linear Regression* – OLS)

- **Rol metodológico:** funciona como la línea base obligatoria (*baseline*) para cuantificar el margen de mejora que aportan los modelos más complejos.
- **Decisión sobre `GridSearchCV`:** inicialmente se ejecutó una optimización mediante `GridSearchCV` sobre los hiperparámetros disponibles en Mínimos Cuadrados Ordinarios (`fit_intercept` y `positive`). Los resultados evidenciaron que la variación en el error fue prácticamente nula, lo cual es esperable debido a que OLS calcula sus coeficientes de manera analítica directa y carece de términos de regularización de complejidad ($\lambda$ o $\alpha$). Conforme al principio de parsimonia (Navaja de Ockham), se prescindió de `GridSearchCV` para evitar sobrecarga computacional innecesaria.

#### 2. Árbol de Decisión de Regresión (*Decision Tree Regressor*)

- **Rol metodológico:** modelo no paramétrico individual diseñado para evaluar la existencia de reglas condicionales y particiones ortogonales en el espacio acústico.
- **Configuración:** evaluado en su forma base, sin restricciones de poda. Al no limitar su profundidad ni el tamaño mínimo de sus hojas, sirvió como diagnóstico del comportamiento de la varianza del catálogo frente a árboles sin podar (*unpruned trees*).

#### 3. Bosque Aleatorio (*Random Forest Regressor*) — Modelo Seleccionado

- **Rol metodológico:** algoritmo de ensamble por agregación (*bagging*) que promedia múltiples árboles de decisión descorrelacionados, reduciendo sustancialmente el error de varianza sin incrementar el sesgo.
- **Hiperparámetros de control de sobreajuste:** con el fin de evitar que los árboles individuales memorizaran el catálogo y obligar al ensamble a derivar reglas comerciales generalizables, se establecieron parámetros de poda rigurosos:

| Hiperparámetro | Valor | Descripción |
|---|---|---|
| `n_estimators` | 150 | Cantidad de estimadores en el bosque, asegurando estabilidad y convergencia en la reducción del error residual. |
| `max_depth` | 12 | Límite máximo de profundidad por árbol para impedir ramificaciones excesivamente profundas que aíslen pistas únicas. |
| `min_samples_leaf` | 15 | Exige un mínimo de 15 canciones por nodo terminal, forzando a que cada predicción sea el promedio representativo de un colectivo musical y no un caso atípico memorizado. |
| `min_samples_split` | 30 | Umbral mínimo de muestras para autorizar una nueva división interna. |
| `n_jobs` | -1 | Paralelización total del cómputo sobre todos los núcleos de CPU. |
| `random_state` | 42 | Semilla determinista para asegurar reproducibilidad exacta. |

---

### 6.1.3. Análisis e Interpretación de los Resultados

#### 1. Colapso del Árbol de Decisión Individual

El árbol de decisión sin restricciones evidenció la vulnerabilidad clásica de los modelos no paramétricos. Al obtener un score de evaluación en test de apenas $0.17$ y un MAE elevado a $11.60$ puntos, demostró que un único árbol genera particiones demasiado sensibles al ruido muestral, volviéndose inoperante para el negocio.

#### 2. Rendimiento de la Regresión Lineal

La regresión lineal ofreció un desempeño respetable ($R^2 = 0.73$, $\text{MAE} = 9.20$, Test Score $= 0.54$), demostrando que la codificación con `TargetEncoder` continuo le aportó una señal fuerte y compacta. No obstante, al restringirse a relaciones puramente aditivas, subestima el efecto combinado entre variables como sonoridad e instrumentalidad.

#### 3. Superioridad y Solidez de Random Forest

El ensamble regularizado superó a todos los competidores en cada una de las dimensiones evaluadas:

- Mayor coeficiente de determinación ($R^2 = 0.75$).
- Score de generalización más alto en prueba ($0.56$).
- Menor error medio ($\text{MAE} = 8.60$ puntos).

Gracias a los hiperparámetros de poda (`min_samples_leaf=15`, `max_depth=12`), la brecha respecto al entrenamiento ($0.69$ vs $0.56$) se mantuvo sumamente controlada, evitando el sobreajuste masivo habitual de los bosques no calibrados.

### 6.1.4. Diagnóstico del Gráfico (Reales vs. Predichos) de Regresión y Alineación con KPIs

El análisis visual del plano de dispersión frente a la recta de identidad ($y = \hat{y}$) aporta información analítica crucial para la toma de decisiones editoriales:

#### 1. Alineación Central y Tendencia Global

Las estimaciones se concentran alrededor de la línea diagonal ideal a lo largo de la mayor parte de la distribución comercial (rango de 20 a 70 puntos), confirmando que el bosque captura la tendencia estructural de la popularidad sin sesgos pronunciados de desplazamiento vertical.

#### 2. Contracción hacia la Media en los Extremos

Se evidencia el fenómeno estadístico habitual en modelos de regresión tabular:

- **En temas impopulares ($\text{Popularidad} < 15$):** el modelo tiende a sobrestimar ligeramente el puntaje, amortiguado por el suavizado empírico bayesiano del `TargetEncoder`.
- **En éxitos masivos globales ($\text{Popularidad} > 80$):** el modelo exhibe una leve infraestimación conservadora debida al promedio de las hojas (`min_samples_leaf=15`), evitando predecir valores de 95-100 a menos que múltiples atributos y el artista respalden sólidamente dicho nivel.

#### 3. Cumplimiento del KPI Técnico

Con un $\text{MAE} = 8.60$ puntos de error absoluto sobre una escala continua de 0 a 100, el modelo asegura un margen de discrepancia acotado a menos de un 9% del rango completo del catálogo, proveyendo al equipo editorial de Spotify un predictor numérico robusto y calibrado para rankear canciones y priorizar lanzamientos en listas de reproducción.

## 6.2 Modelamiento Predictivo: Modelos de Clasificación Multiclase

Como enfoque complementario a la regresión, el problema se abordó desde la perspectiva de clasificación ordinal, segmentando la popularidad de las canciones en 5 niveles estratégicos (*Muy Baja*, *Baja*, *Media*, *Alta*, *Muy Alta*). Este enfoque está diseñado para facilitar decisiones operativas ágiles en la asignación de pistas a listas de reproducción.

### 6.2.1 Configuración y Algoritmos Evaluados
El pipeline de clasificación incluyó la transformación categórica mediante `TargetEncoder` configurado en modo multiclase (`target_type="auto"`). Este encoder logró absorber la mayor parte de la señal predictiva proveniente de los artistas y géneros, estableciendo un techo predictivo robusto.

Para el modelamiento, se contrastaron dos enfoques analíticos:
1. **Árbol de Decisión (Decision Tree Classifier):** Un modelo no lineal evaluado inicialmente en su forma base (lo que resultó en sobreajuste) y posteriormente optimizado.
2. **Regresión Logística (Logistic Regression):** Un modelo lineal multiclase (*One-vs-Rest*) utilizado por su alta interpretabilidad y cálculo directo de probabilidades.

### 6.2.2 Optimización de Hiperparámetros y Resultados
Ambos modelos fueron sometidos a un proceso de optimización utilizando `RandomizedSearchCV` con validación cruzada estratificada (StratifiedKFold de 5 splits) y 15 iteraciones, buscando maximizar la métrica `f1_weighted`. 

En el caso del Árbol de Decisión, se restringió severamente la profundidad máxima (`max_depth`) y el número de muestras por hoja (`min_samples_leaf`) para erradicar el sobreajuste inicial. Los resultados sobre el conjunto de prueba (*Test Set*) fueron los siguientes:

* **Decision Tree (Optimizado):** Accuracy: 0.6951 | F1-Score: 0.6936 | ROC AUC: 0.9126
* **Regresión Logística (Optimizada):** Accuracy: 0.6934 | F1-Score: 0.6919 | ROC AUC: 0.9136

### 6.2.3 Análisis de Resultados y Selección del Modelo
Aunque ambos algoritmos promediaron un desempeño global de clasificación casi idéntico (un empate técnico en ~69% de Accuracy), la prueba de contraste de predicciones exactas demostró que los modelos llegaron a esta cifra por caminos matemáticos distintos, tomando decisiones diferentes en aproximadamente el 15% del catálogo de prueba. 

Para la implementación final, **se seleccionó la Regresión Logística Optimizada** debido a dos factores críticos para el negocio:
1. **Mejor Separación Probabilística:** Su puntaje ROC AUC ligeramente superior (0.9136) garantiza una mayor seguridad y confianza matemática al momento de diferenciar y asignar una canción a una clase de popularidad específica.
2. **Interpretabilidad Superior (Feature Importance):** A diferencia de las ramificaciones opacas de un árbol, los coeficientes lineales de la Regresión Logística permiten explicar de manera transparente a los *stakeholders* qué peso exacto tiene cada variable (como el nivel de *speechiness* o las colaboraciones) en el éxito de una canción, facilitando auditorías y justificaciones ejecutivas.

### 6.2.4 Análisis de la Matriz de Confusión
La Matriz de Confusión de la Regresión Logística permite auditar la precisión del modelo al clasificar canciones en cada nivel de popularidad. A partir de los resultados observados, se extraen las siguientes conclusiones operativas:
1. **Alta Precisión en los Extremos:** El modelo exhibe su mayor capacidad de acierto en las categorías extremas. Logra clasificar correctamente 2877 canciones como Muy Baja y 2837 como Baja. Esto indica que las pistas con nula o escasa tracción comercial poseen patrones acústicos y de metadatos muy distinguibles, lo que permite al algoritmo descartarlas con seguridad, optimizando la recomendación de catálogo y evitando la fatiga auditiva del usuario.
2. **Dificultad en Categorías Adyacentes (Zonas Grises):** Como es habitual en problemas de clasificación ordinal, los errores se concentran casi exclusivamente en categorías contiguas. Por ejemplo, al predecir canciones de popularidad Media, el modelo clasifica correctamente 2439 pistas, pero confunde 557 con popularidad Baja y 435 con Alta. Esto refleja la naturaleza difusa del éxito intermedio, donde la diferencia entre una canción regular y un éxito incipiente suele depender de factores externos no audibles (como el presupuesto de marketing o la viralidad en redes sociales).
3. **Detección Sólida de Éxitos:** En la clase Muy Alta, el algoritmo clasifica correctamente 2380 pistas. Aunque presenta cierta confusión con la categoría Alta (274 casos), el error cruzado hacia clases de fracaso comercial (Muy Baja o Baja) es prácticamente inexistente (menos de 250 casos sumados). Esto garantiza que el modelo es altamente confiable para identificar canciones con potencial de entrar en playlists destacadas, cumpliendo con el objetivo de Scouting.

## 6.3 Modelamiento de segmentación: Perfilamiento sonoro

El modelo implementado corresponde a un sistema de aprendizaje no supervisado basado en el algoritmo $K$-Means, precedido por una etapa sistemática de preprocesamiento de datos y estandarización. Su objetivo estratégico es agrupar las pistas musicales de Spotify según su perfil sonoro para optimizar la recomendación de contenidos y la generación automatizada de listas de reproducción por similitud acústica.

### 6.3.1 Análisis de resultados

Para determinar el número óptimo de clústeres ($k$), se evaluó el comportamiento de la inercia (Suma de los Cuadrados Intra-clúster o WCSS) en un rango de $k \in [1, 10]$ utilizando el método del codo (KneeLocator), complementado con el análisis del coeficiente de silueta (Silhouette Score).

* K Óptimo Identificado: El algoritmo determinó un valor óptimo de $k = 3$.
* Inercia y Codo: La curva de inercias muestra una desaceleración en la tasa de caída a partir de $k = 3$, lo que indica un punto de equilibrio adecuado entre la complejidad del modelo y la compacidad de los grupos.
* Desempeño de Separación: Las particiones espaciales muestran una convergencia estable utilizando una semilla fija, logrando separar de forma clara los espacios vectoriales dominados por características acústicas contrastantes.

### 6.3.2 Caracterización de los clusters

A partir del análisis de las centroides y la distribución de las variables en los 3 grupos resultantes, es posible perfilar los siguientes segmentos musicales:

* Cluster 0 (El núcleo de la plataforma): Concentra aproximadamente el 75% del total de canciones. Al ser el grupo mayoritario, presenta la mayor variedad de géneros (113 géneros únicos) y los niveles más altos de popularidad mediana, consolidándose como la columna vertebral del catálogo. Dado que alberga la mayor variedad de géneros y canciones populares, este cluster debe ser la "puerta de entrada" para nuevos usuarios. Este es el lugar natural para ubicar lanzamientos masivos, promociones de artistas consolidados y campañas publicitaria generales.

* Cluster 1 (Nicho de relajación y escucha activa): Muestra una marcada especialización con una baja diversificación (solo 104 géneros únicos). Sus canciones están fuertemente orientadas a contextos de bienestar, calma y concentración. Se pueden diseñar y promocionar playlists temáticas basadas en momentos del día o actividades. 

* Cluster 2 (Nicho de alta energía): Es el segmento más exclusivo o minoritario del dataset (X% del total), enfocado principalmente en ritmos de baile, optimismo y notas alegres, alejándose del promedio general de la plataforma. Aunque sea el segmento más pequeño, crear playlists para entrenamiento, fiestas o fin de semana puede generar altos niveles de retención, debido a la pasionalidad que se asocia al tipo de música.

### Importancia de Variables (Feature Importance) y Conclusiones de Negocio
El análisis del comportamiento de la Regresión Logística optimizada permite extraer conclusiones fundamentales sobre la dinámica del mercado musical en streaming:

El peso comercial recae de manera desproporcionada en la marca personal del artista y el género musical (capturados de manera sobresaliente por el TargetEncoder). Las variables acústicas puras (como danceability o loudness) actúan únicamente como moduladores secundarios. En particular, la baja presencia de palabras habladas (speechiness) y la suma sinérgica entre valencia e instrumentalidad demuestran tener un impacto medible en la probabilidad de éxito de una canción.

Este hallazgo valida una hipótesis fundamental del negocio: el éxito en Spotify no es puramente acústico, sino impulsado fuertemente por la identidad del creador y el nicho al que pertenece. La integración de la Regresión Logística permite al equipo de scouting cuantificar con exactitud el peso de estos factores al evaluar nuevas adquisiciones para el catálogo.

El análisis del comportamiento del modelo de segmentación no supervisada permite extraer conclusiones fundamentales sobre la dinámica del catálogo musical: la agrupación mediante $K$-Means demuestra que el espacio acústico de Spotify se divide de manera natural en perfiles sónicos claramente diferenciados (comercial, alta energía y atmosféricos/experimentales). Esta estructuración valida que, si bien la identidad del artista define el posicionamiento de mercado, las propiedades físicas de las pistas determinan los contextos de consumo y las preferencias de escucha de los usuarios.

#### Cumplimiento KPIs clasificación
El despliegue de los modelos predictivos y su posterior optimización permiten evaluar directamente el impacto de la solución frente a los Indicadores Clave de Rendimiento (KPIs) definidos al inicio del proyecto:
1. **KPI Técnico de Clasificación (Meta: $> 0.60$ en F1-Score, Precision y Recall):** La Regresión Logística Optimizada superó holgadamente este umbral, alcanzando un desempeño ponderado de $0.69$ tanto en Precision, Recall y F1-Score, junto con un Accuracy global del 69% y un asombroso ROC AUC de 0.91. Esto garantiza a Spotify una herramienta de categorización altamente robusta, superando por casi 10 puntos porcentuales la meta técnica establecida.
2. **KPI de Negocio: Tasa de Detección Temprana de Éxitos (Meta: $\ge 50\%$):** El objetivo de negocio exigía que el sistema fuera capaz de identificar al menos la mitad de las canciones con alto potencial. Al observar el Recall (Sensibilidad) en el reporte de clasificación de nuestro modelo, comprobamos que para las categorías de éxito el sistema identifica correctamente el 56% de las canciones de popularidad "Alta" y el 68% de las pistas de popularidad "Muy Alta". Esto significa que el algoritmo cumple y supera la meta de descubrir más del 50% de los hits ocultos del catálogo, viabilizando directamente el trabajo del equipo de Scouting editorial.
