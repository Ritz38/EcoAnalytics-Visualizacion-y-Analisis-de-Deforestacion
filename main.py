import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def interpolar(df):
    df = df.interpolate(method="linear")
    return df.fillna(method='ffill').fillna(method='bfill')

def leer_datos(url):
    df = pd.read_csv(url)
    return organizar_geometry(interpolar(df))

def organizar_geometry(df):
    df = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_xy(df.Longitud, df.Latitud))
    return df

def propiedades_df(df):
    """Muestra las propiedades principales de un DataFrame de pandas."""
    st.write('Las dimensiones del DataFrame son:', df.shape)
    st.write('\nLos nombres de las columnas son:\n', list(df.columns))
    st.write('\nLos tipos de datos en las columnas son:', df.dtypes)

def estadisticas_vegetacion(df):
    df['Superficie_Total'] = df['Superficie_Deforestada'] / (df['Tasa_Deforestacion'] / 100)

    vegetacion = df.groupby(df['Tipo_Vegetacion'])
    Tasa_Deforestacion = vegetacion[['Superficie_Total','Superficie_Deforestada']].sum()
    Tasa_Deforestacion['Tasa_Deforestacion'] = (Tasa_Deforestacion['Superficie_Deforestada'] / Tasa_Deforestacion['Superficie_Total']) * 100
    st.write('\nLa superficie total, deforestada y el porcentaje de deforestacion por vegetacion es:\n')
    st.write(Tasa_Deforestacion)

    st.write(f'\nEl promedio de deforestacion por vegetacion es:\n')
    st.write(vegetacion['Superficie_Deforestada'].mean())
    
    st.write('\nLa cantidad minima y maxima de deforestación por vegetación es:\n')
    min_values = vegetacion['Superficie_Deforestada'].min()
    max_values = vegetacion['Superficie_Deforestada'].max()
    
    stats_df = pd.DataFrame({'Mínimo': min_values, 'Máximo': max_values})
    st.write(stats_df)

def mapa_por_vegetación(df, base):
    st.title('Mapa por vegetación')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    base.plot(ax=ax, color='white', edgecolor='black')
    df.plot(ax=ax, column="Tipo_Vegetacion", cmap="Set2", legend=True, marker='o', markersize=10)
    st.pyplot(fig)

def mapa_por_altitud(df, base):
    st.title('Mapa por altitud')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    base.plot(ax=ax, color='white', edgecolor='black')
    df.plot(ax=ax, column="Altitud", cmap="coolwarm", legend=True, marker='o', markersize=10)
    st.pyplot(fig)

def mapa_por_precipitacion(df, base):
    st.title('Mapa por precipitacion')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    base.plot(ax=ax, color='white', edgecolor='black')
    df.plot(ax=ax, column="Precipitacion", cmap="coolwarm", legend=True, marker='o', markersize=10)
    st.pyplot(fig)

def mapa_personalizado(df, base):
    st.title('Mapa Personalizado por Variables Seleccionadas')
    
    # Selección de hasta 4 variables
    variables_disponibles = df.columns.tolist()
    selected_variables = st.multiselect("Selecciona hasta 4 variables", variables_disponibles, max_selections=4)
    
    # Filtrar las columnas que contienen texto
    columnas_texto = df[selected_variables].select_dtypes(include=['object']).columns.tolist()
    
    # Inicializar el DataFrame filtrado
    filtered_df = df.copy()

    for variable in selected_variables:
        if variable in columnas_texto:
            # Si la columna contiene texto, se seleccionan los valores únicos
            valores_unicos = df[variable].unique()
            selected_values = st.multiselect(f"Selecciona los valores para {variable}", valores_unicos, default=valores_unicos)
            filtered_df = filtered_df[filtered_df[variable].isin(selected_values)]
        else:
            # Si la columna contiene valores numéricos, se seleccionan los rangos
            min_val, max_val = st.slider(f"Selecciona el rango para {variable}", float(df[variable].min()), float(df[variable].max()), (float(df[variable].min()), float(df[variable].max())))
            filtered_df = filtered_df[(filtered_df[variable] >= min_val) & (filtered_df[variable] <= max_val)]
    
    # Crear el gráfico con los datos filtrados
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    base.plot(ax=ax, color='white', edgecolor='black')
    filtered_df.plot(ax=ax, column=selected_variables[0], cmap="coolwarm", legend=True, marker='o', markersize=10)
    st.pyplot(fig)
    st.pyplot(fig)

def analisis_cluster(df):
    st.title('Análisis de Clúster de Superficies Deforestadas')
    
    # Selección de variables para el análisis de clúster
    variables = ['Superficie_Deforestada', 'Tasa_Deforestacion', 'Altitud', 'Precipitacion']
    selected_variables = st.multiselect('Selecciona las variables para el análisis de clúster', variables, default=variables[:2])
    
    # Estándarización de las variables seleccionadas
    data = df[selected_variables].dropna()
    
    # Normalizar los datos (usamos los rangos de las variables)
    data_scaled = (data - data.min()) / (data.max() - data.min())
    
    # Aplicar linkage (agregación jerárquica de clústeres)
    linked = linkage(data_scaled, method='ward')
    
    # Mostrar dendrograma
    st.write("Dendrograma de los clústeres:")
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(linked, labels=data.index, ax=ax)
    st.pyplot(fig)
    
    # Generar clústeres con un umbral
    threshold = st.slider('Selecciona el umbral para los clústeres', min_value=0, max_value=200, value=100)
    clusters = fcluster(linked, t=threshold, criterion='distance')
    
    # Añadir los clústeres al dataframe
    df['Cluster'] = clusters
    
    # Mostrar los resultados del clúster
    st.write(f"Clústeres creados con las variables: {', '.join(selected_variables)}")
    st.write(df[['Cluster'] + selected_variables].head())
    
    # Mostrar un gráfico de dispersión de los clústeres
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    scatter = ax.scatter(df[selected_variables[0]], df[selected_variables[1]], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel(selected_variables[0])
    ax.set_ylabel(selected_variables[1])
    fig.colorbar(scatter)
    st.pyplot(fig)

def grafico_torta(df):
    st.title('Gráfico de Torta por Tipo de Vegetación')
    
    # Crear un gráfico de torta para mostrar la distribución por tipo de vegetación
    vegetation_count = df['Tipo_Vegetacion'].value_counts()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.pie(vegetation_count, labels=vegetation_count.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.axis('equal')  # Asegura que el gráfico sea un círculo
    st.pyplot(fig)

def interaccion_usuario(df):
    ruta_mapa = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    mundo_dataframe = gpd.read_file(ruta_mapa)
    
    # Filtrar base del mapa para South America
    base = mundo_dataframe[mundo_dataframe['CONTINENT'] == 'South America']

    mapa_por_vegetación(df, base)
    mapa_por_altitud(df, base)
    mapa_por_precipitacion(df, base)
    
    # Llamar a las funciones de mapas y análisis
    mapa_personalizado(df, base)
    analisis_cluster(df)
    grafico_torta(df)

def main():
    # Título de la app
    st.title("Análisis de Datos de Deforestación")

    # Subir archivo o pegar link
    option = st.selectbox('Selecciona cómo quieres cargar los datos', ['Subir archivo', 'Pegar enlace'])
    global df
    df = None
    
    if option == 'Subir archivo':
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=["csv"])
        if uploaded_file is not None:
            df = leer_datos(uploaded_file)
            st.write("Datos cargados correctamente.")
    
    elif option == 'Pegar enlace':
        url = st.text_input('Pega el enlace del archivo CSV:')
        if url:
            df = leer_datos(url)
            st.write("Datos cargados correctamente.")
    
    # Mostrar las propiedades del DataFrame
    if df is not None:
        propiedades_df(df)
        estadisticas_vegetacion(df)
        interaccion_usuario(df)

if __name__ == '__main__':
    main()
