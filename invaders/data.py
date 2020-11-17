import numpy as np

class NeuralData():
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    shape_of_single_frame = (1, (y_max-y_min),(x_max-x_min))

    buffer_len = 1

    frame_buffer = np.zeros(shape_of_single_frame, dtype=int)
    action_buffer = np.zeros(1, dtype=int)
    
    def gray_crop(self, ndarray):
        # Cortando imagem, e convertendo para escala de cinza. Eixos: [y, x]
        return np.mean(ndarray[self.y_min:self.y_max, self.x_min:self.x_max], axis=2).reshape(self.shape_of_single_frame)

    def store_frame_on_buffer(self, frame):
        if len(self.frame_buffer) > self.buffer_len:
            self.frame_buffer = self.frame_buffer[1:-1]
            self.frame_buffer = np.concatenate((self.frame_buffer, frame))
        else:
            self.frame_buffer = np.concatenate((self.frame_buffer, frame))

    def get_frame_buffer_shape(self):
        return self.frame_buffer.shape
    
    def reset_buffer(self):
        self.frame_buffer = np.zeros(self.shape_of_single_frame, dtype=int)
        self.action_buffer = np.zeros(1, dtype=int)

    def store_action_on_buffer(self, action):
        if len(self.action_buffer) > self.buffer_len:
            self.action_buffer = self.action_buffer[1:-1]
            self.action_buffer = np.append(self.action_buffer, np.array(action))
        else:
            self.action_buffer = np.append(self.action_buffer, np.array(action))
    
    
class SupervisionData():
    """
    docstring
    """

    y_min, y_max, x_min, x_max = 25, 195, 20, 140
    shape_of_single_frame = (1, (y_max-y_min),(x_max-x_min))

    match_buffer = {}

    frame_buffer = np.zeros(shape_of_single_frame, dtype=int)
    action_buffer = np.zeros(1, dtype=int) 
    reward_buffer = np.zeros(1, dtype=int)
    life_buffer = np.zeros(1, dtype=int)

    def __init__(self, **kwargs):
        self.PATH = str(kwargs['path'])
        self.NUM_OF_SUPERVISIONS = kwargs['num_of_supervisions']
        self.SAVE_SUPERVISION_DATA_AS_PNG = kwargs['save_supervision_data_as_png']
        self.SAVE_SUPERVISION_DATA_AS_NPZ = kwargs['save_supervision_data_as_npz']
    
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

    def store_match_on_buffer(self, match):
        # Apenas para remover dados desnecessarios de inicializacao
        self.frame_buffer = self.frame_buffer[1:-1]
        self.action_buffer = self.action_buffer[1:-1]
        self.reward_buffer = self.reward_buffer[1:-1]
        self.life_buffer = self.life_buffer[1:-1]
        # Apenas para remover dados desnecessarios de inicializacao

        self.match_buffer.update({match : (self.frame_buffer.shape[0], self.frame_buffer, self.action_buffer, self.reward_buffer, self.life_buffer)})

        # Zerar variaveis
        self.frame_buffer = np.zeros(self.shape_of_single_frame, dtype=int)
        self.action_buffer = np.zeros(1, dtype=int)
        self.reward_buffer = np.zeros(1, dtype=int)
        self.life_buffer = np.zeros(1, dtype=int)
        # Zerar variaveis
    
    def save_supervision_data(self):
        if self.SAVE_SUPERVISION_DATA_AS_PNG:
            self.save_as_png()
        if self.SAVE_SUPERVISION_DATA_AS_NPZ:
            self.save_as_npz()
    
    def save_as_png(self):
        from PIL import Image

        for m in range(self.NUM_OF_SUPERVISIONS):
            num_of_frames = self.match_buffer[m][0]
            array_of_frames = self.match_buffer[m][1]

            for i in range(num_of_frames):
                img = Image.fromarray(array_of_frames[i])
                img = img.convert("L")

                path = self.PATH + "img/match_" + str(m) + "/" + str(i) + ".png"
                img.save(path)
        
        print("Successfully saved as PNG.")

    def save_as_npz(self):
        for m in range(self.NUM_OF_SUPERVISIONS):
            np.savez_compressed(self.PATH + "npz/match_" + str(m) + "/frames.npz", self.match_buffer[m][1])
            np.savez_compressed(self.PATH + "npz/match_" + str(m) + "/actions.npz", self.match_buffer[m][2])
            np.savez_compressed(self.PATH + "npz/match_" + str(m) + "/rewards.npz", self.match_buffer[m][3])
            np.savez_compressed(self.PATH + "npz/match_" + str(m) + "/lifes.npz", self.match_buffer[m][4])
    
        print("Successfully saved as NPZ.")

    def load_npz(self):
        import os

        for folder in os.listdir(self.PATH + "npz/"):
            match = int(folder.split("_")[1])
            
            actions = np.load(self.PATH + "npz/" + str(folder) + '/actions.npz')
            lifes = np.load(self.PATH + "npz/" + str(folder) + '/lifes.npz')
            frames = np.load(self.PATH + "npz/" + str(folder) + '/frames.npz')
            rewards = np.load(self.PATH + "npz/" + str(folder) + '/rewards.npz')

            array_of_actions = actions.f.arr_0
            array_of_lifes = lifes.f.arr_0
            array_of_frames = frames.f.arr_0
            array_of_rewards = rewards.f.arr_0

            self.match_buffer.update({match : (array_of_frames.shape[0], array_of_frames, array_of_actions, array_of_rewards, array_of_lifes)})
        
        print("Successfully loaded NPZ.")