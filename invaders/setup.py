from game import Invaders

PATH = "../../.database/pibic/invaders/"

MATCHES = 300
NUM_OF_EPOCHS = 1

LOAD_MODEL = False
VERBOSE = True

SUPERVISION = True
LOAD_SUPERVISION_DATA = True
SAVE_SUPERVISION_DATA_AS_PNG = False
SAVE_SUPERVISION_DATA_AS_NPZ = False
NUM_OF_SUPERVISIONS = 5
# SLEEP = 0.07
SLEEP = 0

SELF_TRAIN = False
RENDER_SELF_TRAIN = False
NUM_OF_TRAINS = 30


invaders = Invaders(
        path=PATH,
        matches=MATCHES,
        trains=NUM_OF_TRAINS,
        epochs=NUM_OF_EPOCHS,
        verbose=VERBOSE,
        render_self_train=RENDER_SELF_TRAIN,
        supervision=SUPERVISION,
        num_of_supervisions=NUM_OF_SUPERVISIONS,
        load_supervision_data=LOAD_SUPERVISION_DATA,
        save_supervision_data_as_png=SAVE_SUPERVISION_DATA_AS_PNG,
        save_supervision_data_as_npz=SAVE_SUPERVISION_DATA_AS_NPZ,
        load_model=LOAD_MODEL,
        sleep=SLEEP,
    )

if LOAD_MODEL:
    invaders.simplernn.load()
else:
    if SUPERVISION:
        invaders.run_supervision_training()
    if SELF_TRAIN:
        invaders.run_self_training()

invaders.run_predict()