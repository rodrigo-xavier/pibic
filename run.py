# from space_invaders import Play
from space_invaders import PlayLearning, PlayAutomatic, PlayManual

# Escolher o numero de partidas que o jogo devera ter
MATCHES = 10000



# play = PlayLearning()
play = PlayAutomatic()
# play = PlayManual()

play.run(MATCHES)
# play.run_storing(MATCHES)

exit()