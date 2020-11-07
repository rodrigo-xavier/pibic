from game import Invaders

PATH = "../../.database/pibic/invaders/"

MATCHES = 500
NUM_OF_TRAINS = 30
NUM_OF_EPOCHS = 3

VERBOSE = True
RENDER_TRAIN = True

invaders = Invaders(
        path=PATH,
        matches=MATCHES,
        trains=NUM_OF_TRAINS,
        epochs=NUM_OF_EPOCHS,
        verbose=VERBOSE,
        render=RENDER_TRAIN,
    )
invaders.run()