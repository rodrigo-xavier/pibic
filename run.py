from space_invaders import Play

RUN_CHOICE= {
    "AUTOMATIC":    1,
    "MANUAL":       2,
}

play = Play()
play.go(50, RUN_CHOICE["AUTOMATIC"])
exit()