# roblox-up-and-coming-forecast-01 v3

A make.fun Roblox Up-and-Coming chart forecaster: a Markov drift / transition model that learns how game ranks move and propagates the latest chart forward. Node 22, standard library only.

## Run

`mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts` — reads the case history, writes forecasts to the output directory.
