from game import Invaders

PATH = "../../.database/pibic/invaders/"
MATCHES = 500

invaders = Invaders(PATH, MATCHES)
invaders.run()