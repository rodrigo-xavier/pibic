# from space_invaders import Play
from space_invaders import Play

# Escolher se a nave devera ser controlada manualmente, pelo proprio ambiente do gym ou por uma IA
RUN_CHOICE= {
    "AUTOMATIC":    1,
    "MANUAL":       2,
    "DEEPLEARNING": 3,
}
# Escolher o numero de partidas que o jogo devera ter
MATCHES = 50



play = Play()
play.go(MATCHES, RUN_CHOICE["DEEPLEARNING"])
exit()