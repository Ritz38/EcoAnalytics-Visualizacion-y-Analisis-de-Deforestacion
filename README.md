# EcoAnalytics: Visualización y Análisis de Deforestación

## Descripción del Proyecto

**EcoAnalytics** es una herramienta interactiva diseñada para analizar y visualizar datos relacionados con la deforestación en América del Sur. La aplicación permite a los usuarios cargar datos en formato CSV, ya sea subiendo un archivo o proporcionando un enlace, y ofrece una variedad de funcionalidades para explorar y entender los patrones de deforestación en diferentes tipos de vegetación, altitudes y niveles de precipitación.

**CSV SUGERIDO:** https://drive.google.com/file/d/1c4txVXyo40OgT0hSrwplLjlsjUX7rlZs/view?usp=sharing

## Características principales

### 1. Carga de Datos
- Los usuarios pueden cargar datos desde un archivo CSV o mediante un enlace.
- Los datos se procesan para asegurar que no haya valores faltantes y se organizan en un formato geoespacial.

### 2. Análisis de Vegetación
- Calcula la superficie total, la superficie deforestada y la tasa de deforestación por tipo de vegetación.
- Muestra estadísticas como el promedio, mínimo y máximo de deforestación por tipo de vegetación.

### 3. Visualización de Mapas
- **Mapa por Vegetación**: Visualiza la distribución de los tipos de vegetación en un mapa.
- **Mapa por Altitud**: Muestra la distribución de la deforestación en función de la altitud.
- **Mapa por Precipitación**: Representa la deforestación en relación con los niveles de precipitación.
- **Mapa Personalizado**: Permite a los usuarios seleccionar hasta 4 variables para crear un mapa personalizado, aplicando filtros según los valores de las variables seleccionadas.

### 4. Análisis de Clúster
- Realiza un análisis de clúster jerárquico para agrupar áreas con patrones similares de deforestación.
- Muestra un dendrograma y permite ajustar el umbral para la creación de clústeres.
- Visualiza los clústeres en un gráfico de dispersión.

### 5. Gráfico de Torta
- Muestra la distribución de los tipos de vegetación en un gráfico de torta, proporcionando una visión general de la composición de la vegetación en los datos.

### 6. Interfaz de Usuario
- La aplicación está construida con **Streamlit**, lo que permite una interacción sencilla y amigable con los datos.
- Los usuarios pueden explorar los datos de manera dinámica, aplicando filtros y visualizando los resultados en tiempo real.

## Tecnologías Utilizadas
- **Python**: Lenguaje de programación principal.
- **Pandas**: Para la manipulación y análisis de datos.
- **GeoPandas**: Para el manejo de datos geoespaciales.
- **Matplotlib**: Para la creación de gráficos y mapas.
- **Streamlit**: Para la creación de la interfaz de usuario interactiva.
- **SciPy**: Para el análisis de clúster jerárquico.

## Aplicaciones Potenciales
- **Investigación Ambiental**: Para estudiar los patrones de deforestación y su relación con factores como la altitud y la precipitación.
- **Toma de Decisiones**: Para ayudar a los responsables de políticas a identificar áreas críticas de deforestación y planificar estrategias de conservación.
- **Educación**: Como herramienta educativa para enseñar conceptos de análisis de datos y visualización en el contexto de la conservación ambiental.

**EcoAnalytics** es una herramienta poderosa para cualquier persona interesada en el análisis de datos ambientales, ofreciendo una combinación de análisis estadístico y visualización geoespacial en una interfaz fácil de usar.
