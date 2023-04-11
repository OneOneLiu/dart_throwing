import numpy as np
import matplotlib.pyplot as plt
import glob

q_tables = glob.glob('q_table_*reward*')
print(q_tables)
q_table = np.load('q_table_0_reward_-20.0.npy')

# Create the initial figure and image
fig, ax = plt.subplots()
im = ax.imshow(q_table, cmap='viridis')

for q_table in q_tables:
    # Update the image data
    ax.set_title(q_table)
    q_table = np.load(q_table)
    im.set_data(q_table)
    # Redraw the image on the plot
    fig.canvas.draw()
    
    # Pause for a short time to allow the image to be displayed
    plt.pause(5)