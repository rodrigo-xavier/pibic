from game import Invaders

PATH = "../../.database/pibic/invaders/"

MATCHES = 3000
NUM_OF_EPOCHS = 10
HIDDEN_NEURONS = 200

MAKE_REINFORCEMENT_DATA = False
LOAD_MODEL = False
VERBOSE = True

SLEEP = 0.06
# SLEEP = 0
# [0:132]

invaders = Invaders(
        path=PATH,
        matches=MATCHES,
        epochs=NUM_OF_EPOCHS,
        hidden_neurons=HIDDEN_NEURONS,
        verbose=VERBOSE,
        load_model=LOAD_MODEL,
        sleep=SLEEP,
    )


if LOAD_MODEL:
    invaders.simplernn.load()
else:
    if MAKE_REINFORCEMENT_DATA:
        invaders.save_reinforcement()
    else:
        invaders.load_reinforcement_and_train()
        invaders.test_overfitting()

invaders.run_predict()



# DON'T USE THIS, SERIOUSLY
def gamb_split_new_array():
    num_of_frames, frames, actions, rewards, lifes = invaders.reinforcement.load_npz(0)
    invaders.reinforcement.reset()
    invaders.reinforcement.match_buffer = [133, frames[0:132], actions[0:132], rewards[0:132], lifes[0:132]]
    invaders.reinforcement.save_as_npz()

gamb_split_new_array()