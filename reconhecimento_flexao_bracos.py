# coding=utf-8

import cv2
import numpy as np

# Caminhos das estruturas da rede neural pré-treinados
arquivo_proto = "./pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
arquivo_pesos = "./pose/body/mpi/pose_iter_160000.caffemodel"

# Pontos utilizados no Modelo MPII
numero_pontos = 15
pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14],
                [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

# Dimensões do video da entrada da rede
entrada_largura = 256
entrada_altura = 256

y_linha_base_pulso = 0
y_linha_base_cotovelo = None
percentual_movimento_flexao = 0
qtd_flexao_valida = 0

qtd_frames = 0
pontos = None

# Carrega o Video que será analisado.
# video_entrada = "./videos/flexao_bracos_lateral_perto.mp4"
video_entrada = "./videos/flexao_bracos_frontal.mp4"
# video_entrada = "./videos/flexao_bracos_lateral_distante.mp4"


# Acessa o video que será analisado..
captura = cv2.VideoCapture(video_entrada)
_, frame_inicial = captura.read()

# Define as configurações do vídeo de saida que vai conter os pontos detectados e movimentos detectados.
video_saida = "./output/flexao_bracos_saida.avi"
gravar_video = cv2.VideoWriter(video_saida, cv2.VideoWriter_fourcc(*'XVID'), 10,
                               (frame_inicial.shape[1], frame_inicial.shape[0]))

# Carrega a RN que já foi treinada e os pesos.
modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)
# Define o limite mínimo do mapa de confiança
limite_confianca = 0.3  # 30%


def desenha_pontos(numero_pontos, saida_RN, video_largura, video_altura):
    pontos = []
    altura = saida_RN.shape[2]
    largura = saida_RN.shape[3]

    for i in range(numero_pontos):
        mapa_confianca = saida_RN[0, i, :, :]
        _, confianca, _, ponto = cv2.minMaxLoc(mapa_confianca)

        x = (video_largura * ponto[0]) / largura
        y = (video_altura * ponto[1] / altura)

        if confianca > limite_confianca:
            cv2.circle(video_copia, (int(x), int(y)), 4, (255, 0, 128), thickness=3, lineType=cv2.FILLED)
            cv2.putText(video_copia, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            pontos.append((int(x), int(y)))
        else:
            pontos.append(None)

    return pontos


def desenha_tracos(pontos, pares_pontos):
    for par in pares_pontos:
        inicio_traco = par[0]
        fim_traco = par[1]

        if pontos[inicio_traco] and pontos[fim_traco]:
            cv2.line(frame, pontos[inicio_traco], pontos[fim_traco], (192, 200, 192), 2, lineType=cv2.LINE_AA)
            cv2.line(video_copia, pontos[inicio_traco], pontos[fim_traco], (192, 200, 192), 2, lineType=cv2.LINE_AA)

            cv2.circle(frame, pontos[inicio_traco], 4, (14, 201, 255), thickness=1, lineType=cv2.FILLED)
            cv2.circle(frame, pontos[fim_traco], 4, (14, 201, 255), thickness=1, lineType=cv2.FILLED)


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
    cv2.putText(frame_saida, " linha base pulso", inicio, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    return y_linha_base


def desenha_linha_base_cotovelo(pontos, frame_saida, video_largura, y_linha_base, y_linha_base_pulso):
    largura_frame = int(video_largura)

    cotevelo_direito = pontos[3]
    cotovelo_esquerdo = pontos[6]

    if y_linha_base is None:
        y_linha_base = y_linha_base_pulso + 1

    if cotevelo_direito and (cotevelo_direito[1] < y_linha_base and cotevelo_direito[1] < y_linha_base_pulso):
        y_linha_base = cotevelo_direito[1]

    if cotovelo_esquerdo and (cotovelo_esquerdo[1] < y_linha_base and cotovelo_esquerdo[1] < y_linha_base_pulso):
        y_linha_base = cotovelo_esquerdo[1]

    inicio = (0, y_linha_base)
    fim = (largura_frame, y_linha_base)
    cv2.line(frame_saida, inicio, fim, (0, 0, 255), 1)
    cv2.putText(frame_saida, " linha base cotovelo", inicio, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

    return y_linha_base


def desenha_linha_limite_movimentos_validos(frame_saida, y_linha_base_cotovelo, largura_frame):
    limite_movimento_superior = y_linha_base_cotovelo - int(y_linha_base_cotovelo * .4)
    limite_movimento_inferior = y_linha_base_cotovelo + int(y_linha_base_cotovelo * .15)

    cv2.line(frame_saida, (0, limite_movimento_superior), (int(largura_frame), limite_movimento_superior), (255, 0, 0),
             1)
    cv2.line(frame_saida, (0, limite_movimento_inferior), (int(largura_frame), limite_movimento_inferior), (255, 0, 0),
             1)

    cv2.putText(frame_saida, " limite cabeca superior", (0, limite_movimento_superior), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 255, 0))
    cv2.putText(frame_saida, " limite cabeca inferior", (0, limite_movimento_inferior), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 255, 0))

    return limite_movimento_inferior, limite_movimento_superior


def checa_flexao(pontos, percentual_movimento_flexao, limite_movimento_inferior, limite_movimento_superior,
                 y_linha_base_cotovelo):
    cabeca = pontos[0]

    if cabeca:

        y_cabeca = cabeca[1]
        # 25% do movimento completo
        if percentual_movimento_flexao == 0 and (y_cabeca < y_linha_base_pulso and y_cabeca < y_linha_base_cotovelo):
            percentual_movimento_flexao = 0.25

        # 50% do movimento completo
        if percentual_movimento_flexao == 0.25 and (
                y_cabeca >= limite_movimento_inferior and y_cabeca >= y_linha_base_cotovelo):
            percentual_movimento_flexao = 0.5

        # 75% do movimento completo
        if percentual_movimento_flexao == 0.5 and (
                y_cabeca <= limite_movimento_inferior and y_cabeca <= y_linha_base_cotovelo):
            percentual_movimento_flexao = 0.75

        # 100% do movimento completo
        if percentual_movimento_flexao == 0.75 and (
                y_cabeca <= y_linha_base_cotovelo and y_cabeca <= limite_movimento_superior):
            percentual_movimento_flexao = 1

    return percentual_movimento_flexao


while (True):

    conectado, frame = captura.read()
    video_copia = np.copy(frame)
    video_largura = frame.shape[1]
    video_altura = frame.shape[0]

    qtd_frames += 1

    # Processa somente os frames onde a quatidade for multiplo de 3 para agilizar o processamento.
    if qtd_frames % 3 == 0:
        # Conversão do tipo da imagem para o tipo utilizado pela rede
        blob_entrada = cv2.dnn.blobFromImage(frame, 1.0 / 255, (entrada_largura, entrada_altura), (0, 0, 0),
                                             swapRB=False,
                                             crop=False)

        modelo.setInput(blob_entrada)
        saida_rede = modelo.forward()

        # Desenha os pontos detectados na imagem.
        pontos = desenha_pontos(numero_pontos, saida_rede, video_largura, video_altura)

        # Desenha os traços conectando os pontos detectados na imagem.
        desenha_tracos(pontos, pares_pontos)

    # desenha_linha_cotovelos(pontos, video_copia, video_largura)

    if pontos is not None:

        # Desenha a linha de base do pulso que será usada como referência do "chão"
        y_linha_base_pulso = desenha_linha_base_pulso(pontos, video_copia, video_largura, y_linha_base_pulso)

        # Desenha a linha de base do cotovelo que será usada como referência para validação do movimento.
        y_linha_base_cotovelo = desenha_linha_base_cotovelo(pontos, video_copia, video_largura, y_linha_base_cotovelo,
                                                            y_linha_base_pulso)

        if y_linha_base_cotovelo and y_linha_base_pulso:
            # Desenha a linha que define o movimento válido.
            limite_movimento_inferior, limite_movimento_superior = desenha_linha_limite_movimentos_validos(video_copia,
                                                                                                           y_linha_base_cotovelo,
                                                                                                           video_largura)

            # Verfica se o movimento realizado é um movimento válido.
            percentual_movimento_flexao = checa_flexao(pontos,
                                                       percentual_movimento_flexao,
                                                       limite_movimento_inferior,
                                                       limite_movimento_superior,
                                                       y_linha_base_cotovelo
                                                       )

        cv2.putText(video_copia, "Movimento {:.0%} concluido".format(percentual_movimento_flexao), (10, 35),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        if percentual_movimento_flexao == 1:
            qtd_flexao_valida += 1
            percentual_movimento_flexao = 0

        cv2.putText(video_copia, "Movimentos Validos: {}".format(qtd_flexao_valida), (10, 20), cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 255, 0))

    # Exibe o Video
    cv2.imshow('Video', video_copia)

    # Salva um cópia do frame
    gravar_video.write(video_copia)

    # Aperte a tecla 'q' para sair.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos.
gravar_video.release()
cv2.destroyAllWindows()
