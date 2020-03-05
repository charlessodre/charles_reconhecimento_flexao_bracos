import time
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Caminhos das estruturas da rede neural pré-treinados
arquivo_proto = "./pose/body/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
arquivo_pesos = "./pose/body/mpi/pose_iter_160000.caffemodel"

# Modelo MPII
numero_pontos = 15
pares_pontos = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],[1,14],
               [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]

# Define as cores do ponto em BGR
cor_pontoA, cor_pontoB, cor_linha = (14, 201, 255), (255, 0, 128), (192, 192, 192)
cor_txtponto, cor_txtinicial, cor_txtandamento = (10, 216, 245), (255, 0, 128), (192, 192, 192)

# Define formato da fonte do texto
tamanho_fonte, tamanho_linha, tamanho_circulo, espessura = 0.7, 2, 5, 1
fonte = cv2.FONT_HERSHEY_SIMPLEX


valida_pernas_juntas, valida_pernas_afastadas = 0, 0
valida_bracos_abaixo, valida_bracos_acima = 0, 0

entrada_largura = 256
entrada_altura = 256


# Carrega o Video que será analisado.
frame = "./videos/flexao_bracos_reduzido.mp4"
captura = cv2.VideoCapture(frame)
conectado, frame = captura.read()

# Salva o frame com os pontos detectados.
video_saida = "./flexao_bracos_saida.avi"
gravar_video = cv2.VideoWriter(video_saida, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame.shape[1], frame.shape[0]))

# Carrega o modelo que já foi treinado.
modelo = cv2.dnn.readNetFromCaffe(arquivo_proto, arquivo_pesos)
limite = 0.1 # 10%


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
            cv2.circle(video_copia, (int(x), int(y)), 4, cor_pontoB,
                       thickness=tamanho_circulo,
                       lineType=cv2.FILLED)
            cv2.putText(video_copia, "{}".format(i), (int(x), int(y)),
                        fonte, tamanho_fonte, cor_txtponto, 3,
                        lineType=cv2.LINE_AA)
            cv2.putText(fundo, " ", (int(x), int(y)),
                        fonte, tamanho_fonte, cor_txtponto, 3,
                        lineType=cv2.LINE_AA)

            pontos.append((int(x), int(y)))
        else:
            pontos.append(None)

    return pontos

def desenha_tracos(pontos, pares_pontos):

    for par in pares_pontos:
        parteA = par[0]
        parteB = par[1]

        if pontos[parteA] and pontos[parteB]:
            cv2.line(frame, pontos[parteA], pontos[parteB], cor_linha,
                     tamanho_linha, lineType=cv2.LINE_AA)
            cv2.line(video_copia, pontos[parteA], pontos[parteB], cor_linha,
                     tamanho_linha, lineType=cv2.LINE_AA)
            cv2.line(fundo, pontos[parteA], pontos[parteB], cor_linha,
                     tamanho_linha, lineType=cv2.LINE_AA)

            cv2.circle(frame, pontos[parteA], 4, cor_pontoA, thickness=espessura,
                       lineType=cv2.FILLED)
            cv2.circle(frame, pontos[parteB], 4, cor_pontoA, thickness=espessura,
                       lineType=cv2.FILLED)
            cv2.circle(fundo, pontos[parteA], 4, cor_pontoA, thickness=espessura,
                       lineType=cv2.FILLED)
            cv2.circle(fundo, pontos[parteB], 4, cor_pontoA, thickness=espessura,
                       lineType=cv2.FILLED)

def desenha_linha_cotovelos(pontos):


    #video_largura = frame.shape[1]
    #video_altura = frame.shape[0]

    #al = int(video_altura / 2)
    lag = int(video_largura)
    #inicio = (0, al)
    #fim = (lag, al)
    #cv2.line(video_copia, inicio, fim, (0, 0, 255), tamanho_linha, lineType=cv2.LINE_AA)

    if pontos[3] or pontos[6]:
        if pontos[3]:
            inicio = (0, pontos[3][1])
            fim = (lag, pontos[3][1])
        if pontos[6]:
            inicio = (0, pontos[6][1])
            fim = (lag, pontos[6][1])

        cv2.line(video_copia, inicio, fim, (0, 0, 255), tamanho_linha, lineType=cv2.LINE_AA)






process_this_frame = True

while (True):
    t = time.time()
    conectado, frame = captura.read()
    video_copia = np.copy(frame)




    # Only process every other frame of video to save time
    if process_this_frame:

        video_largura = frame.shape[1]
        video_altura = frame.shape[0]

        # Criação da máscara com fundo preto
        tamanho = cv2.resize(frame, (video_largura, video_altura))
        mapa_suave = cv2.GaussianBlur(tamanho, (3, 3), 0, 0)
        fundo = np.uint8(mapa_suave > limite)

        # Conversão do tipo da imagem
        blob_entrada = cv2.dnn.blobFromImage(frame, 1.0 / 255, (entrada_largura, entrada_altura), (0, 0, 0), swapRB=False, crop=False)

        modelo.setInput(blob_entrada)
        saida_rede = modelo.forward()

        pontos = desenha_pontos(numero_pontos, saida_rede, video_largura, video_altura)

        desenha_tracos(pontos, pares_pontos)

        # TODO teste
        desenha_linha_cotovelos(pontos)
        # TODO teste

    process_this_frame = not process_this_frame

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