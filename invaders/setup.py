from game import Invaders

PATH = "../../.database/pibic/invaders/"

MATCHES = 500
NUM_OF_EPOCHS = 3

VERBOSE = True

SUPERVISION = True
LOAD_SUPERVISION_DATA = False
SAVE_SUPERVISION_DATA_AS_PNG = False
SAVE_SUPERVISION_DATA_AS_NPZ = True
NUM_OF_SUPERVISIONS = 2

TRAIN = False
RENDER_TRAIN = True
NUM_OF_TRAINS = 30



invaders = Invaders(
        path=PATH,
        matches=MATCHES,
        trains=NUM_OF_TRAINS,
        epochs=NUM_OF_EPOCHS,
        verbose=VERBOSE,
        render_train=RENDER_TRAIN,
        num_of_supervisions=NUM_OF_SUPERVISIONS,
        load_supervision_data=LOAD_SUPERVISION_DATA,
        save_supervision_data_as_png=SAVE_SUPERVISION_DATA_AS_PNG,
        save_supervision_data_as_npz=SAVE_SUPERVISION_DATA_AS_NPZ,
    )


if SUPERVISION:
    invaders.run_supervision_training()
if TRAIN:
    invaders.run_self_training()

invaders.run_predict()