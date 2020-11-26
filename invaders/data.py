import numpy as np
import os

class NeuralData():
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    shape_of_single_frame = (1, (y_max-y_min),(x_max-x_min))

    match_buffer = []
    buffer_len = 30

    frame_buffer = np.zeros(shape_of_single_frame, dtype=int)
    action_buffer = np.zeros(1, dtype=int) 
    reward_buffer = np.zeros(1, dtype=int)
    life_buffer = np.zeros(1, dtype=int)

    def __init__(self, **kwargs):
        self.PATH = str(kwargs['path'])
        self.prepare_folder_to_save_data()
    
    def prepare_folder_to_save_data(self):
        m = []
        listdir = os.listdir(self.PATH + "npz/")

        if listdir:
            for folder in listdir:
                m.append(int(folder.split("_")[1]))
            number_of_last_folder = max(m)
            self.next_folder = "match_" + str(number_of_last_folder + 1)
        else:
            self.next_folder = "match_0"

        os.mkdir(self.next_folder)
    
    def gray_crop(self, ndarray):
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2).reshape(self.shape_of_single_frame)

    def get_last_action(self):
        return self.action_buffer[-1]
    
    def store_data_on_buffer(self, frame, reward, life, action):
        frame = self.gray_crop(frame)

        self.frame_buffer = np.concatenate((self.frame_buffer, frame))
        self.reward_buffer = np.append(self.reward_buffer, np.array(reward))
        self.life_buffer = np.append(self.life_buffer, np.array(life))
        self.action_buffer = np.append(self.action_buffer, np.array(action))

    def save_data(self):
        # Apenas para remover dados desnecessarios de inicializacao
        self.frame_buffer = self.frame_buffer[1:-1]
        self.action_buffer = self.action_buffer[1:-1]
        self.reward_buffer = self.reward_buffer[1:-1]
        self.life_buffer = self.life_buffer[1:-1]
        # Apenas para remover dados desnecessarios de inicializacao

        self.match_buffer = [self.action_buffer.shape[0], self.frame_buffer, self.action_buffer, self.reward_buffer, self.life_buffer]
    
        if input('Do you want to save data as npz (y/n)? ')=='y':
            self.save_as_npz()
        if input('Do you want to save data as png (y/n)? ')=='y':
            self.save_as_png()
    
    def save_as_npz(self):
        path = self.PATH + "npz/" + str(self.next_folder) + "/"

        np.savez_compressed(path + "frames.npz", self.match_buffer[1])
        np.savez_compressed(path + "actions.npz", self.match_buffer[2])
        np.savez_compressed(path + "rewards.npz", self.match_buffer[3])
        np.savez_compressed(path + "lifes.npz", self.match_buffer[4])
    
        print("Successfully saved as NPZ.")

    def save_as_png(self):
        from PIL import Image

        _path = self.PATH + "img/" + str(self.next_folder) + "/"

        num_of_frames = self.match_buffer[0]
        array_of_frames = self.match_buffer[1]

        for i in range(num_of_frames):
            img = Image.fromarray(array_of_frames[i])
            img = img.convert("L")
            path = _path + str(i) + ".png"
            img.save(path)
    
        print("Successfully saved as PNG.")
    
    def num_of_samples(self):
        return len(os.listdir(self.PATH + "npz/"))

    def load_npz(self, m):
        path = self.PATH + "npz/match_" + str(m) + "/"
        
        actions = np.load(path + 'actions.npz')
        lifes = np.load(path + 'lifes.npz')
        frames = np.load(path + 'frames.npz')
        rewards = np.load(path + 'rewards.npz')

        arr_actions = actions.f.arr_0
        arr_lifes = lifes.f.arr_0
        arr_frames = frames.f.arr_0
        arr_rewards = rewards.f.arr_0

        self.match_buffer = [arr_actions.shape[0], arr_frames, arr_actions, arr_rewards, arr_lifes]
        
        print("Successfully loaded NPZ.")