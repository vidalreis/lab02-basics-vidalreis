import numpy as np
from PIL import Image
import streamlit as st
from skimage.transform import resize

# Funções de manipulação de imagem
def aplicar_escala_cinza(imagem, estrategia='media'):
    if estrategia == 'media':
        return np.mean(imagem, axis=2)
    elif estrategia == 'luminancia':
        return np.dot(imagem[...,:3], [0.299, 0.587, 0.114])
    elif estrategia == 'linear':
        return np.dot(imagem[...,:3], [0.2126, 0.7152, 0.0722])
    else:
        raise ValueError("Estratégia inválida: use 'media', 'luminancia' ou 'linear'")

def ampliar_imagem(imagem, fator=2, interpolacao='vizinho'):
    altura, largura = imagem.shape[:2]
    nova_altura, nova_largura = int(altura * fator), int(largura * fator)

    if interpolacao == 'vizinho':
        resultado = np.zeros((nova_altura, nova_largura) + (() if imagem.ndim == 2 else (3,)))
        for y in range(nova_altura):
            for x in range(nova_largura):
                orig_y = min(int(y / fator), altura - 1)
                orig_x = min(int(x / fator), largura - 1)
                resultado[y, x] = imagem[orig_y, orig_x]
    else:  # bilinear
        resultado = resize(imagem, (nova_altura, nova_largura), order=1, mode='reflect')

    return resultado

def reduzir_imagem(imagem, fator=0.5, estrategia='sem_filtro'):
    altura, largura = imagem.shape[:2]
    nova_altura, nova_largura = int(altura * fator), int(largura * fator)

    if estrategia == 'sem_filtro':
        passo = int(1 / fator)
        return imagem[::passo, ::passo]
    else:  # média
        resultado = np.zeros((nova_altura, nova_largura) + (() if imagem.ndim == 2 else (3,)))
        for y in range(nova_altura):
            for x in range(nova_largura):
                bloco = imagem[int(y/fator):int((y+1)/fator), int(x/fator):int((x+1)/fator)]
                if imagem.ndim == 2:
                    resultado[y, x] = np.mean(bloco)
                else:
                    resultado[y, x] = np.mean(bloco, axis=(0, 1))
        return resultado

# Interface do aplicativo Streamlit
st.sidebar.title("Menu Lateral")
arquivo = st.sidebar.file_uploader("Selecione uma imagem", type=["jpg", "jpeg", "png"])

st.title("Editor de Imagens")

if arquivo is not None:
    imagem_original = Image.open(arquivo)
    imagem_np = np.array(imagem_original).astype('float32') / 255.0

    aba1, aba2, aba3 = st.tabs(["Cores", "Preto & Branco", "Redimensionar"])

    with aba1:
        st.subheader("Ajuste de Canais RGB")
        r = st.slider("Vermelho", 0.0, 2.0, 1.0, 0.01)
        g = st.slider("Verde", 0.0, 2.0, 1.0, 0.01)
        b = st.slider("Azul", 0.0, 2.0, 1.0, 0.01)

        imagem_colorida = imagem_np.copy()
        imagem_colorida[:, :, 0] *= r
        imagem_colorida[:, :, 1] *= g
        imagem_colorida[:, :, 2] *= b
        imagem_colorida = np.clip(imagem_colorida, 0, 1)

        st.image(imagem_colorida, caption="Imagem com Cores Ajustadas", clamp=True, use_container_width=True)

    with aba2:
        st.subheader("Conversão para Escala de Cinza")
        escolha = st.radio("Escolha o método:", ['Média', 'Luminância', 'Linear'], horizontal=True)
        metodos = {'Média': 'media', 'Luminância': 'luminancia', 'Linear': 'linear'}
        imagem_pb = aplicar_escala_cinza(imagem_np, estrategia=metodos[escolha])

        imagem_pb_rgb = np.dstack([imagem_pb] * 3)
        st.image(imagem_pb_rgb, caption="Imagem em Escala de Cinza", clamp=True, use_container_width=True)

    with aba3:
        st.subheader("Ajuste de Tamanho")
        col_a, col_b = st.columns(2)
        with col_a:
            acao = st.radio("Tipo:", ['Aumentar', 'Reduzir'], horizontal=True)
        with col_b:
            if acao == 'Aumentar':
                escala = st.slider("Escala", 1.1, 4.0, 2.0, 0.1)
                metodo = st.radio("Interpolação:", ['Vizinho Próximo', 'Bilinear'], horizontal=True)
                metodo = 'vizinho' if metodo == 'Vizinho Próximo' else 'bilinear'
            else:
                escala = st.slider("Escala", 0.1, 0.9, 0.5, 0.1)
                metodo = st.radio("Redução:", ['Sem Filtro', 'Média'], horizontal=True)
                metodo = 'sem_filtro' if metodo == 'Sem Filtro' else 'media'

        if acao == 'Aumentar':
            imagem_redimensionada = ampliar_imagem(imagem_np, fator=escala, interpolacao=metodo)
        else:
            imagem_redimensionada = reduzir_imagem(imagem_np, fator=escala, estrategia=metodo)

        st.image(imagem_redimensionada, caption=f"Imagem {acao}", clamp=True, use_container_width=True)

else:
    st.info("Carregue uma imagem para começar.")
