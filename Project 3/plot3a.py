import numpy as np
import matplotlib.pyplot as plt

def plot3a(u1, u2, u3, u4, x_len, y_len):
    u1 = u1.reshape(   (x_len, x_len)  ) 
    u2 = u2.reshape(   (y_len, x_len)  )
    u3 = u3.reshape(   (x_len, x_len)  )
    u4 = u4.reshape(   (int(x_len/2), int(x_len/2))  )
    
    # we can debug with random values like this
    # u1 = np.random.rand(20, 20)*20
    # u2 = np.random.rand(40, 20)*50
    # u3 = np.random.rand(20, 20)*70

    room_length = np.size(u1,1)

    # Create an empty matrix to store all temperatures, considering the layout:
    total_temp = np.zeros((room_length * 2, room_length * 3))

    # Assign each room's temperature to the appropriate slice of the total matrix:
    # total_temp[y, x]
    total_temp[room_length:, :room_length] = u1  # Bottom left
    total_temp[:, room_length:room_length*2] = u2   # Middle
    total_temp[:room_length, room_length*2:] = u3   # Top right
    b1 = int(room_length*3/2)
    b2 = int(room_length*5/2)
    total_temp[room_length:b1, room_length*2:b2] = u4

    # this removes zeros entries that are outside any room
    masked_temp = np.ma.masked_where(total_temp == 0, total_temp)

    # Visualize
    plt.imshow(masked_temp, cmap='hot', origin='lower', vmin=0, vmax=50)
    plt.colorbar(label='Temperature')
    plt.title('Temperature Distribution in Rooms')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.gca().invert_yaxis()
    plt.show()