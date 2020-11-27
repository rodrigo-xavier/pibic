from game import Invaders

PATH = "../../.database/pibic/invaders/"

MATCHES = 3000
SLEEP = 0.06

MAKE_REINFORCEMENT_DATA = False
LOAD_MODEL = False
VERBOSE = True

for neurons in range(140, 301, 10):
    for epochs in range(0, 101, 20):
        NUM_OF_EPOCHS = epochs
        HIDDEN_NEURONS = neurons

        invaders = Invaders(
                path=PATH,
                matches=MATCHES,
                epochs=NUM_OF_EPOCHS,
                hidden_neurons=HIDDEN_NEURONS,
                verbose=VERBOSE,
                load_model=LOAD_MODEL,
                sleep=SLEEP,
            )

        invaders.load_reinforcement_and_train()
        invaders.test_overfitting()
        
        del invaders