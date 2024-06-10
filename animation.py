import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

leg_length = 1  # Length of the leg segments
foot_length = 0.2  # Length of the foot segment

# Function to update the animation every frame
def update(frame, emg_data, grf_data, y_true, y_pred, lines):
    update_leg(lines['true_leg'], y_true[frame])  # Update true leg lines
    update_leg(lines['pred_leg'], y_pred[frame])  # Update predicted leg lines

    for muscle, line in lines['emg'].items():
        line.set_data(np.arange(frame + 1), emg_data[muscle][:frame + 1])  # Update EMG data for each muscle

    for axis, line in lines['grf'].items():
        line.set_data(np.arange(frame + 1), grf_data[axis][:frame + 1])

    # lines['true_angle'].set_data(np.arange(frame + 1), y_true[:frame + 1])
    # lines['pred_angle'].set_data(np.arange(frame + 1), y_pred[:frame + 1])
    
    return [lines['true_leg'], lines['pred_leg']] + list(lines['emg'].values()) + list(lines['grf'].values())

# Function to update leg segments
def update_leg(line, knee_angle):
    hip = np.array([0, 0])  # Hip is always at the origin
    knee = hip + np.array([0, -1])  # Knee always 1 unit below the hip

    if (knee_angle < 0):
        knee_angle = 0

    ankle_y = - 1 - (leg_length * np.cos(np.deg2rad(knee_angle)))
    ankle_x = - (leg_length * np.sin(np.deg2rad(knee_angle)))

    ankle = np.array([ankle_x, ankle_y])
    
    foot_x = np.sin(np.deg2rad(90-knee_angle)) * foot_length
    foot_y = -(np.cos(np.deg2rad(90-knee_angle)) * foot_length)

    foot = ankle + np.array([foot_x, foot_y])

    # Update row data
    line.set_data([hip[0], knee[0], ankle[0], foot[0]], [hip[1], knee[1], ankle[1], foot[1]])

# Main function to create the animation
def create_animation(best_model, emg_data, grf_data, y_true, y_pred):
    # Check if arrays have the same length
    if len(y_true) != len(y_pred):
        raise ValueError("The y_true and y_pred arrays must have the same length.")
    
    frames = len(y_true)  # The number of frames is the length of the y_true array (samples)

    # Create the figure and axes
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 10))

    # Configure the leg chart
    ax1.set_xlim(-1, 2)
    ax1.set_ylim(-3, 1)
    true_leg, = ax1.plot([], [], 'b-', lw=2, label='Real')
    pred_leg, = ax1.plot([], [], 'r--', lw=2, label='Predicted')
    ax1.set_aspect('equal')
    ax1.legend()

    # Configure the EMG signal graph
    ax2.set_xlim(0, frames)
    ax2.set_ylim(-1, np.max([np.max(emg_data[muscle]) for muscle in emg_data]) * 1.1)  # Add top margin for EMG
    ax2.set_xlabel('Samples in time (250Hz)')
    ax2.set_ylabel('EMG [% of Maximum Voluntary Contraction]')
    emg_lines = {muscle: ax2.plot([], [], label=muscle)[0] for muscle in emg_data}
    ax2.legend()

    # Configure the grf signal graph
    ax3.set_xlim(0, frames)
    ax3.set_ylim(np.min([np.min(grf_data[axis]) for axis in grf_data]) * 1.1, np.max([np.max(grf_data[axis]) for axis in grf_data]) * 1.1)  # Add top margin for GRF
    ax3.set_xlabel('Samples in time (250Hz)')
    ax3.set_ylabel('Ground Reaction Forces [N]')
    grf_lines = {axis: ax3.plot([], [], label=axis)[0] for axis in grf_data}
    ax3.legend()


    # Dictionary to pass the lines to the update function
    lines = {'true_leg': true_leg, 'pred_leg': pred_leg, 'emg': emg_lines, 'grf': grf_lines}


    # Create the animation
    anim = FuncAnimation(fig, update, frames=frames, fargs=(emg_data, grf_data, y_true, y_pred, lines), interval=0.5)

    # plt.suptitle(f'Prediction of angle of knee joint with {best_model}')
    plt.suptitle(f'Knee Angle Prediction', fontsize=20)
    plt.show()


# Example of how to call the function with random data
if __name__ == "__main__":
    # Gerar dados de exemplo (substitua pelos seus dados reais)
    frames = 1001

    emg_anim = {
        'BF': np.random.rand(frames),
        'VL': np.random.rand(frames),
        'TA': np.random.rand(frames),
        'GAL': np.random.rand(frames)
    }

    y_true = np.linspace(0, 90, frames)  # Angulação real do joelho de 0 a 90 graus
    y_pred = y_true + np.random.normal(0, 5, frames)  # Angulação predita com algum ruído

    # Exibir informações sobre os dados gerados
    print("Dados EMG:")
    for muscle, data in emg_anim.items():
        print(f"{muscle}: {data.shape}, {type(data)}")

    print("\nDados y_true e y_pred:")
    print(y_true.shape, type(y_true))
    print(y_pred.shape, type(y_pred))

    # Run the animation with the best model    
    create_animation("Test", emg_anim, y_true, y_pred)