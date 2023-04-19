import streamlit as st
from skimage import io
from matplotlib.pyplot import imshow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import graycomatrix, graycoprops
import matplotlib.ticker as ticker

st.set_page_config(page_title="Image Processing", page_icon="🖼️")
st.title("Welcome to Image Processing! 👋")
st.caption("Image Processing is a web-based application that can analyze the colors and textures of images based on various techniques. This app allows users to analyze more than 4,000 image samples for color analysis (such as converting an image to a pixel matrix, generating a color histogram, and calculating first-order statistics) and texture analysis (such as generating the Gray Level Co-occurrence Matrix (GLCM) of an image and creating a texture histogram). This app was created by a [student](https://github.com/agungmahadana/) using Python and Streamlit.")

# Backend
def get_images(absen, clas):
    all_class = ['a', 'b', 'c', 'd']
    start = 300 * (all_class.index(clas)) + (10 * (absen - 1)) + 1
    end = start + 9
    return list(range(start, end + 1))

def load_image(type, imageKey):
    return io.imread(f'images/FacialExpression/{type}/{type}-{imageKey:04d}.jpg')

def color_histogram(image):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(image.ravel(), bins=255, color='red', alpha=0.7, rwidth=0.85)
    ax.set_title('Colour Histogram', fontweight='bold', fontsize=16)
    ax.set_xlabel('Colour Distribution')
    ax.set_ylabel('Count')
    return fig

def first_order(image):
    mean = np.mean(image)
    variance = np.var(image)
    skewness = skew(np.reshape(image, (48 * 48)))
    kurtos = kurtosis(np.reshape(image, (48 * 48)))
    entrop = entropy(np.reshape(image, (48 * 48)))
    return mean, variance, skewness, kurtos, entrop

def compute_glcm(image, angles):
    glcm = graycomatrix(image, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)
    return glcm

def glcm_matrix(image, metric_texture):
    matrix = []
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for i in metric_texture:
        row = []
        for j in angles:
            row.append(graycoprops(compute_glcm(image, [j]), prop=i)[0][0])
        matrix.append(row)
    return matrix

def texture_histogram(glcm_matrix, metric_texture):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Texture Histogram', fontweight='bold', fontsize=16)
    ax = ax.ravel()
    for i, a in enumerate(ax):
        a.hist(glcm_matrix[i], bins=255, color='red', alpha=0.7, rwidth=0.85)
        a.set_title(metric_texture[i])
        a.grid()
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        a.xaxis.set_major_formatter(formatter)
    return fig

# Frontend
class_input = st.radio("Select your class", ('A', 'B', 'C', 'D'), horizontal=True, index=1)
absen_input = st.slider("Select your absen", 1, 20, value=12)
type_input = st.selectbox("Select image type", ('Happy', 'Sad', 'Neutral'))
image_index_input = st.number_input("Select image index", min_value=1, max_value=10)
image = load_image(type_input.lower(), get_images(absen_input, class_input.lower())[image_index_input - 1])

st.subheader("Image Sample")
col1, col2, col3 = st.columns(3)
with col1:
    st.write('')
with col2:
    st.image(image, caption=f'{type_input.lower()}-{get_images(absen_input, class_input.lower())[image_index_input - 1]:04d}.jpg', use_column_width=True)
with col3:
    st.write('')

st.subheader("Image Matrix")
st.dataframe(image)

st.subheader("Color Histogram")
st.pyplot(color_histogram(image))

st.subheader("First Order Statistics")
first_order_statistics = pd.DataFrame(first_order(image), index=['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Entropy'], columns=['Value'])
st.dataframe(first_order_statistics, use_container_width=True)

st.subheader("Gray Level Co-occurrence Matrix (GLCM)")
metric_texture = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
glcm_matrix = glcm_matrix(image, metric_texture)
glcm = pd.DataFrame(glcm_matrix, index=['Dissimilarity', 'Correlation', 'Homogeneity', 'Contrast', 'ASM', 'Energy'], columns=[0, 45, 90, 135])
st.dataframe(glcm, use_container_width=True)

st.subheader("Texture Histogram")
st.pyplot(texture_histogram(glcm_matrix, metric_texture))