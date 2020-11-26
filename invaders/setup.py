from game import Invaders

PATH = "../../.database/pibic/invaders/"

MATCHES = 3000
NUM_OF_EPOCHS = 10

TEST_OVERFITTING = True
LOAD_MODEL = False
VERBOSE = True

MAKE_REINFORCEMENT_DATA = False
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

if TEST_OVERFITTING:
    invaders.test_overfitting()
else:
    if LOAD_MODEL:
        invaders.simplernn.load()
    else:
        if MAKE_REINFORCEMENT_DATA:
            invaders.save_reinforcement()
        else:
            invaders.load_reinforcement_and_train()

    invaders.run_predict()