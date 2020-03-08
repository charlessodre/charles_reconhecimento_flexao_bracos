import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Caminhos das estruturas da rede neural pré-treinados
arquivo_proto = "./pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
arquivo_pesos = "./pose/body/mpi/pose_iter_160000.caffemodel"

# Ponto utilizado no Modelo MPII
numero_pontos = 15
pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
                [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

# Define as cores do ponto em BGR
cor_pontoA = (14, 201, 255)
cor_pontoB = (255, 0, 128)
cor_linha_esqueleto = (192, 192, 192)

# Define formato da fonte do texto
tamanho_fonte = 1
tamanho_linha = 2
tamanho_circulo = 3
espessura = 1
fonte = cv2.FONT_HERSHEY_PLAIN
cor_fonte = (0, 255, 0)  # BGR

entrada_largura = 256
entrada_altura = 256

y_linha_base_pulso = 0
y_linha_base_cotovelo = None
y_linha_base_cabeca = None
percentual_movimento_flexao = 0
qtd_flexao_valida = 0

qtd_frames = 0
pontos = None

# Carrega o Video que será analisado.
frame = "./videos/flexao_bracos_reduzido.mp4"
captura = cv2.VideoCapture(frame)
conectado, frame = captura.read()

# Salva o frame com os pontos detectados.
video_saida = "./flexao_bracos_saida.avi"
gravar_video = cv2.VideoWriter(video_saida, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame.shape[1], frame.shape[0]))

# Carrega o modelo que já foi treinado.
modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)
limite = 0.2  # 30%


def desenha_pontos(numero_pontos, saida_RN, video_largura, video_altura):
    pontos = []
    altura = saida_RN.shape[2]
    largura = saida_RN.shape[3]

    for i in range(numero_pontos):
        mapa_confianca = saida_RN[0, i, :, :]
        _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)

        x = (video_largura * ponto[0]) / largura
        y = (video_altura * ponto[1] / altura)

        if confianca > limite:
            cv2.circle(video_copia, (int(x), int(y)), 4, cor_pontoB, thickness=tamanho_circulo, lineType=cv2.FILLED)
            cv2.putText(video_copia, "{}".format(i), (int(x), int(y)), fonte, tamanho_fonte, cor_fonte)

            pontos.append((int(x), int(y)))
        else:
            pontos.append(None)

    return pontos


def desenha_tracos(pontos, pares_pontos):
    for par in pares_pontos:
        parteA = par[0]
        parteB = par[1]

        if pontos[parteA] and pontos[parteB]:
            cv2.line(frame, pontos[parteA], pontos[parteB], cor_linha_esqueleto, tamanho_linha, lineType=cv2.LINE_AA)
            cv2.line(video_copia, pontos[parteA], pontos[parteB], cor_linha_esqueleto, tamanho_linha,
                     lineType=cv2.LINE_AA)

            cv2.circle(frame, pontos[parteA], 4, cor_pontoA, thickness=espessura, lineType=cv2.FILLED)
            cv2.circle(frame, pontos[parteB], 4, cor_pontoA, thickness=espessura, lineType=cv2.FILLED)


def desenha_linha_cotovelos(pontos, frame_saida, video_largura):
    largura_frame = int(video_largura)

    # if pontos[0]:
    #     y = pontos[0][1]
    #     inicio = (0, y)
    #     fim = (largura_frame, y)
    #     cv2.line(frame_saida, inicio, fim, (0, 255, 255), 1)
    #     cv2.putText(frame_saida, "cabeca [{}]".format(inicio), inicio, fonte, tamanho_fonte, cor_fonte)

    if pontos[3]:
        y = pontos[3][1]
        inicio = (0, y)
        fim = (largura_frame, y)
        cv2.line(frame_saida, inicio, fim, (255, 0, 255), 1)
        cv2.putText(frame_saida, "cotovelo direito", inicio, fonte, tamanho_fonte, cor_fonte)

    if pontos[6]:
        y = pontos[6][1]
        inicio = (0, y)
        fim = (largura_frame, y)
        cv2.line(frame_saida, inicio, fim, (0, 0, 255), 1)
        cv2.putText(frame_saida, "cotovelo esquerdo", inicio, fonte, tamanho_fonte, cor_fonte)

    # if pontos[4]:
    #     y = pontos[4][1]
    #     inicio = (0, y)
    #     fim = (largura_frame, y)
    #     cv2.line(frame_saida, inicio, fim, (0, 0, 255), 1)
    #     cv2.putText(frame_saida, "pulso direito", inicio, fonte, tamanho_fonte, cor_fonte)
    #
    # if pontos[7]:
    #     y = pontos[7][1]
    #     inicio = (0, y)
    #     fim = (largura_frame, y)
    #     cv2.line(frame_saida, inicio, fim, (0, 0, 255), 1)
    #     cv2.putText(frame_saida, "pulso esquerdo", inicio, fonte, tamanho_fonte, cor_fonte)


def desenha_linha_base_pulso(pontos, frame_saida, video_largura, y_linha_base):
    largura_frame = int(video_largura)

    pulso_direito = pontos[4]
    pulso_esquedo = pontos[7]

    if pulso_direito or pulso_esquedo:
        if pulso_direito and pulso_direito[1] > y_linha_base:
            y_linha_base = pulso_direito[1]

        if pulso_esquedo and pulso_esquedo[1] > y_linha_base:
            y_linha_base = pulso_esquedo[1]

    inicio = (0, y_linha_base)
    fim = (largura_frame, y_linha_base)
    cv2.line(frame_saida, inicio, fim, (0, 0, 255), 1)
    cv2.putText(frame_saida, "linha base pulso", inicio, fonte, tamanho_fonte, cor_fonte)

    return y_linha_base


def desenha_linha_base_cabeca(pontos, frame_saida, video_largura, y_linha_base):
    largura_frame = int(video_largura)

    cabeca = pontos[0]

    if y_linha_base is None:
        y_linha_base = 10000

    if cabeca and cabeca[1] > y_linha_base:
        y_linha_base = cabeca[1]

    inicio = (0, y_linha_base)
    fim = (largura_frame, y_linha_base)
    cv2.line(frame_saida, inicio, fim, (0, 0, 255), 1)
    cv2.putText(frame_saida, "linha base cabeca", inicio, fonte, tamanho_fonte, cor_fonte)

    return y_linha_base


def desenha_linha_base_cotovelo(pontos, frame_saida, video_largura, y_linha_base, y_linha_base_pulso):
    largura_frame = int(video_largura)

    cotevelo_direito = pontos[3]
    cotovelo_esquerdo = pontos[6]

    if y_linha_base is None:
        y_linha_base = y_linha_base_pulso + 1

    if y_linha_base > y_linha_base_pulso:
        if cotevelo_direito or cotovelo_esquerdo:
            if cotevelo_direito and cotevelo_direito[1] > y_linha_base:
                y_linha_base = cotevelo_direito[1]

            if cotovelo_esquerdo and cotovelo_esquerdo[1] > y_linha_base:
                y_linha_base = cotovelo_esquerdo[1]

    inicio = (0, y_linha_base)
    fim = (largura_frame, y_linha_base)
    cv2.line(frame_saida, inicio, fim, (0, 0, 255), 1)
    cv2.putText(frame_saida, "linha base cotovelo", inicio, fonte, tamanho_fonte, cor_fonte)

    return y_linha_base


def checa_flexao(pontos, frame_saida, y_linha_base_pulso, linha_base_cotovelo, percentual_movimento_flexao,
                 qtd_flexao_valida):
    cabeca = pontos[0]

    if cabeca:

        y_cabeca = cabeca[1]
        # 50% do movimento completo
        if y_cabeca < y_linha_base_pulso and y_cabeca > linha_base_cotovelo:
            percentual_movimento_flexao = 0.50
            cv2.putText(frame_saida, "Movimento {:.0%} concluido".format(percentual_movimento_flexao), (300, 35), fonte,
                        tamanho_fonte, cor_fonte)

        # 75% do movimento completo
        if percentual_movimento_flexao == 0.5 and y_cabeca < y_linha_base_pulso and y_cabeca < linha_base_cotovelo:
            percentual_movimento_flexao = 0.75
            cv2.putText(frame_saida, "Movimento {:.0%} concluido".format(percentual_movimento_flexao), (300, 35), fonte,
                        tamanho_fonte, cor_fonte)

        # 100% do movimento completo
        if percentual_movimento_flexao == 0.75 and y_cabeca * 1.1 < y_linha_base_pulso and y_cabeca * 1.1 < linha_base_cotovelo:
            percentual_movimento_flexao = 1
            cv2.putText(frame_saida,
                        "Movimento {:.0%} concluido - {}".format(percentual_movimento_flexao, (cabeca[1] * 1.1)), (
                            300, 35), fonte,
                        tamanho_fonte, cor_fonte)

        if percentual_movimento_flexao == 1:
            qtd_flexao_valida += 1
            percentual_movimento_flexao = 0

    cv2.putText(frame_saida, "Movimentos Validos: {}".format(qtd_flexao_valida), (300, 20), fonte, tamanho_fonte,
                cor_fonte)

    return percentual_movimento_flexao, qtd_flexao_valida


while (True):

    conectado, frame = captura.read()
    video_copia = np.copy(frame)
    video_largura = frame.shape[1]
    video_altura = frame.shape[0]

    qtd_frames += 1
    # Processa somente os frames onde a quatidade for multiplo de 5
    if qtd_frames % 5 == 0:
        # Conversão do tipo da imagem
        blob_entrada = cv2.dnn.blobFromImage(frame, 1.0 / 255, (entrada_largura, entrada_altura), (0, 0, 0),
                                             swapRB=False,
                                             crop=False)

        modelo.setInput(blob_entrada)
        saida_rede = modelo.forward()

        pontos = desenha_pontos(numero_pontos, saida_rede, video_largura, video_altura)

        # desenha_tracos(pontos, pares_pontos)

        # desenha_linha_cotovelos(pontos, video_copia, video_largura)

    if pontos is not None:
        y_linha_base_pulso = desenha_linha_base_pulso(pontos, video_copia, video_largura, y_linha_base_pulso)
        y_linha_base_cotovelo = desenha_linha_base_cotovelo(pontos, video_copia, video_largura, y_linha_base_cotovelo,
                                                            y_linha_base_pulso)

        y_linha_base_cabeca = desenha_linha_base_cabeca(pontos, video_copia, video_largura, y_linha_base_cabeca)

        if y_linha_base_cotovelo and y_linha_base_pulso:
            percentual_movimento_flexao, qtd_flexao_valida = checa_flexao(pontos, video_copia, y_linha_base_pulso,
                                                                          y_linha_base_cotovelo,
                                                                          percentual_movimento_flexao,
                                                                          qtd_flexao_valida)

    # Exibe o Video
    cv2.imshow('Video', video_copia)

    # Salva um cópia do frame
    gravar_video.write(video_copia)

    # Aperte a tecla 'q' para sair.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
gravar_video.release()
cv2.destroyAllWindows()
