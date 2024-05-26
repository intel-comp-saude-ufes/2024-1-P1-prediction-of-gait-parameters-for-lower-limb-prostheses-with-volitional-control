import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função para atualizar a animação a cada quadro
def update(frame, emg_data, y_true, y_pred, lines):
    # Atualizar as linhas da perna verdadeira
    update_leg(lines['true_leg'], y_true[frame])
    # Atualizar as linhas da perna predita
    update_leg(lines['pred_leg'], y_pred[frame])
    # Atualizar os sinais EMG
    lines['emg'].set_data(np.arange(frame + 1), emg_data[:frame + 1])
    # Retornar a lista de objetos atualizados
    return lines['true_leg'], lines['pred_leg'], lines['emg']

# Função para atualizar os segmentos da perna
def update_leg(line, knee_angle):
    hip = np.array([0, 0])
    knee = hip + np.array([0, -1])  # Joelho sempre 1 unidade abaixo do quadril
    ankle = knee + np.array([np.sin(np.deg2rad(-knee_angle)), -np.cos(np.deg2rad(-knee_angle))])

    foot = ankle + np.array([0.05, 0])  # O pé é representado como uma linha horizontal de 0.05 unidade

    # Atualiza os dados da linha
    line.set_data([hip[0], knee[0], ankle[0], foot[0]], [hip[1], knee[1], ankle[1], foot[1]])

# Função principal para criar a animação
def create_animation(emg_data, y_true, y_pred):
    # Verificar se os arrays têm o mesmo comprimento
    if len(y_true) != len(y_pred) or len(y_true) != len(emg_data):
        raise ValueError("Os arrays y_true, y_pred e emg_data devem ter o mesmo comprimento.")

    frames = len(y_true)

    # Criar a figura e os eixos
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Configurar o gráfico das pernas
    ax1.set_xlim(-1, 2)  # Ajuste dos limites do eixo x
    ax1.set_ylim(-3, 1)  # Ajuste dos limites do eixo y
    true_leg, = ax1.plot([], [], 'b-', lw=2, label='Real')
    pred_leg, = ax1.plot([], [], 'r--', lw=2, label='Predito')
    ax1.legend()

    # Configurar o gráfico de sinais EMG
    ax2.set_xlim(0, frames)
    ax2.set_ylim(-np.max(emg_data) * 1.1, np.max(emg_data) * 1.1)  # Adicionar margem superior para EMG
    emg_line, = ax2.plot([], [], 'g-')

    # Dicionário para passar as linhas para a função de atualização
    lines = {'true_leg': true_leg, 'pred_leg': pred_leg, 'emg': emg_line}

    # Criar a animação
    anim = FuncAnimation(fig, update, frames=frames, fargs=(emg_data, y_true, y_pred, lines), interval=5)

    # Mostrar a animação
    plt.show()

# Exemplo de como chamar a função com dados gerados
if __name__ == "__main__":
    # Gerar dados de exemplo (substitua pelos seus dados reais)
    frames = 1001
    emg_data = np.random.rand(frames, 1)  # Exemplo de dados EMG
    print(emg_data)
    print(emg_data.shape)
    print(type(emg_data))

    y_true = np.linspace(0, 90, frames)  # Angulação real do joelho de 0 a 90 graus
    print(y_true)
    print(y_true.shape)
    print(type(y_true))

    y_pred = y_true + np.random.normal(0, 5, frames)  # Angulação predita com algum ruído
    print(y_pred)
    print(y_pred.shape)
    print(type(y_pred))
    
    create_animation(emg_data, y_true, y_pred)