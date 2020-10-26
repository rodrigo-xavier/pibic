from game import Invaders

PATH = "../../.database/pibic/invaders/"
MATCHES = 5

invaders = Invaders(PATH, MATCHES)
invaders.run()