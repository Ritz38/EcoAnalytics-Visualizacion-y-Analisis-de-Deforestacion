import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import streamlit as st

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

def mapa_por_vegetación(ax):
    df.plot(ax=ax, column="Tipo_Vegetacion", cmap="Set2", legend=True, marker='o', markersize=10)

def mapa_por_altitud(ax):
    df.plot(ax=ax, column="Altitud", cmap="coolwarm", legend=True, marker='o', markersize=10)

def mapa_por_precipitacion(ax):
    df.plot(ax=ax, column="Precipitacion", cmap="coolwarm", legend=True, marker='o', markersize=10)

def interaccion_usuario(df):
    ruta_mapa = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"
    mundo_dataframe = gpd.read_file(ruta_mapa)
    
    # Crear figura con 3 subgráficos (1 fila, 3 columnas)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Filtrar base del mapa para South America
    base = mundo_dataframe[mundo_dataframe['CONTINENT'] == 'South America']
    base.plot(ax=axes[0], color='white', edgecolor='black')
    base.plot(ax=axes[1], color='white', edgecolor='black')
    base.plot(ax=axes[2], color='white', edgecolor='black')

    # Mapas por vegetación, altitud y precipitaciones
    mapa_por_vegetación(axes[0])
    mapa_por_altitud(axes[1])
    mapa_por_precipitacion(axes[2])

    # Ajustar el espacio entre subgráficos
    st.pyplot(fig)

def main():
    # Título de la app
    st.title("Análisis de Datos de Deforestación")

    # Subir archivo o pegar link
    option = st.selectbox('Selecciona cómo quieres cargar los datos', ['Subir archivo', 'Pegar enlace'])
    
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
