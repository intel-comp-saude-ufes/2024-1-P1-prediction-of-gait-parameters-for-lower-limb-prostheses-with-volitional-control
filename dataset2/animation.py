import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Function to update the animation every frame
def update(frame, emg_data, y_true, y_pred, lines):
    update_leg(lines['true_leg'], y_true[frame]) # Update true leg lines
    update_leg(lines['pred_leg'], y_pred[frame]) # Update predicted leg lines
    lines['emg'].set_data(np.arange(frame + 1), emg_data[:frame + 1]) # Update EMG data
    return lines['true_leg'], lines['pred_leg'], lines['emg'] # Return the updated lines


# Function to update leg segments
def update_leg(line, knee_angle):
    hip = np.array([0, 0]) # Hip is always at the origin
    knee = hip + np.array([0, -1]) # Knee always 1 unit below the hip
    ankle = knee + np.array([np.sin(np.deg2rad(-knee_angle)), -np.cos(np.deg2rad(-knee_angle))])
    # FIXME: Make leg stop to increase when knee angle is increasing

    # The foot is represented as a horizontal line of 0.05 units
    foot = ankle + np.array([0.05, 0]) 

    # Update row data
    line.set_data([hip[0], knee[0], ankle[0], foot[0]], [hip[1], knee[1], ankle[1], foot[1]])


# Main function to create the animation
def create_animation(best_model, emg_data, y_true, y_pred):
    # Check if arrays have the same length
    if len(y_true) != len(y_pred) or len(y_true) != len(emg_data):
        raise ValueError("The y_true, y_pred and emg_data arrays must have the same length.")

    frames = len(y_true) # The number of frames is the length of the y_true array (samples)

    # Create the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Configure the leg chart
    ax1.set_xlim(-1, 2)
    ax1.set_ylim(-3, 1)
    true_leg, = ax1.plot([], [], 'b-', lw=2, label='Real')
    pred_leg, = ax1.plot([], [], 'r--', lw=2, label='Predito')
    ax1.legend()

    # Configure the EMG signal graph
    ax2.set_xlim(0, frames)
    ax2.set_ylim(-1, np.max(emg_data) * 1.1) # Add top margin for EMG
    emg_line, = ax2.plot([], [], 'g-')

    # Dictionary to pass the lines to the update function
    lines = {'true_leg': true_leg, 'pred_leg': pred_leg, 'emg': emg_line}

    # Create the animation
    anim = FuncAnimation(fig, update, frames=frames, fargs=(emg_data, y_true, y_pred, lines), interval=5)

    plt.suptitle(f'Prediction of angle of knee joint with {best_model}')
    plt.show()


# Example of how to call the function with random data
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