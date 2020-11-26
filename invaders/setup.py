from game import Invaders

PATH = "../../.database/pibic/invaders/"

MATCHES = 3000
NUM_OF_EPOCHS = 1

TEST_OVERFITING = False
LOAD_MODEL = False
VERBOSE = False

LOAD_REINFORCEMENT_DATA = True
NUM_OF_REINFORCEMENTS = 1
SLEEP = 0.06
# SLEEP = 0


invaders = Invaders(
        path=PATH,
        matches=MATCHES,
        epochs=NUM_OF_EPOCHS,
        verbose=VERBOSE,
        num_of_reinforcements=NUM_OF_REINFORCEMENTS,
        load_model=LOAD_MODEL,
        sleep=SLEEP,
    )

if TEST_OVERFITING:
    invaders.test_overfiting()
else:
    if LOAD_MODEL:
        invaders.simplernn.load()
    else:
        if LOAD_REINFORCEMENT_DATA:
            invaders.load_reinforcement_and_train()
        else:
            invaders.save_reinforcement()

    invaders.run_predict()