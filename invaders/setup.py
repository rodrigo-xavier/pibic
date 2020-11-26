from game import Invaders

PATH = "../../.database/pibic/invaders/"

MATCHES = 3000
NUM_OF_EPOCHS = 10
LOAD_MODEL = False
VERBOSE = True

MAKE_REINFORCEMENT_DATA = True
SLEEP = 0.06
# SLEEP = 0
# [0:132]

invaders = Invaders(
        path=PATH,
        matches=MATCHES,
        epochs=NUM_OF_EPOCHS,
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

invaders.run_predict()