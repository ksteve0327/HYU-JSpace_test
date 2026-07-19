# Reading Experiment B — internal reasoning (gemma4b)

- prompt: 'The number of legs on the animal that spins webs is'
- position: last token; concept 'spider' single-token ids: [32261, 40046, 48587, 86757]
- model greedy next token: ' '

| layer | lens top-6 | rank of 'spider' |
|---|---|---|
| 0 | '<start_of_image>' '   ' '\\\\' '}\\' '    ' '  ' | 89207 |
| 1 | '<start_of_image>' '   ' '\\\\' '  ' '\\' '}\\' | 89590 |
| 2 | '   ' '\\\\' '<start_of_image>' '.\\\\' '  ' '}\\' | 83087 |
| 3 | '\\\\' '   ' '<start_of_image>' '.\\\\' ',\\\\' '  ' | 70836 |
| 4 | '.\\\\' '<start_of_image>' '\\\\' '   ' '.\\' '".' | 66214 |
| 5 | '\\\\' '   ' '<start_of_image>' '.\\\\' '  ' ',\\\\' | 85145 |
| 6 | '  ' '   ' '<start_of_image>' '</b>' '.\\\\' '\\\\' | 36748 |
| 7 | '<start_of_image>' '  ' '   ' '.\\\\' '\\\\' '....' | 24216 |
| 8 | '<start_of_image>' '   ' '  ' '.\\\\' '\\\\' '....' | 12981 |
| 9 | '  ' '   ' '<start_of_image>' '.\\\\' '\\\\' '    ' | 7944 |
| 10 | '  ' '   ' '<start_of_image>' '.\\\\' '�' '~\\' | 4055 |
| 11 | '  ' '<start_of_image>' '   ' '.\\\\' '~\\' '.\\' | 2932 |
| 12 | '  ' '   ' '<start_of_image>' '</strong>' '~\\' '�' | 1537 |
| 13 | '  ' '.\\\\' '<start_of_image>' '   ' ':\\\\' '!\\' | 4169 |
| 14 | '  ' '<start_of_image>' '.\\\\' ' fishes' '!' '   ' | 3322 |
| 15 | ' critters' ' beetles' ' insects' ' bunnies' ' turtles' ' giraffe' | 1920 |
| 16 | ' ____' ' critters' ' _____' '_____' ' turtles' '____' | 1852 |
| 17 | ' critters' ' ____' ' squirrels' ' beetles' ' turtles' ' _____' | 1162 |
| 18 | ' ____' ' squirrels' ' _____' ' beetles' ' turtles' ' critters' | 1554 |
| 19 | ' ____' ' giraffe' ' squirrels' ' furry' ' vertebrates' ' crocodiles' | 1064 |
| 20 | ' ____' ' ______' '_____' '____' ' ________' ' _____' | 1298 |
| 21 | ' ____' ' ________' ' ______' ' _______' '____' ' ____________' | 566 |
| 22 | ' ____' '______' ' ________' ' ______' ' _______' '____' | 399 |
| 23 | ' ____' '_____' ' ________' '______' ' _______' ' ______' | 590 |
| 24 | ' ____' ' ________' '______' ' ______' ' _______' '_____' | 81 |
| 25 | ' ____' ' ________' ' _______' ' __________' ' ______' ' _____' | 79 |
| 26 | ' spiders' ' ________' ' ____' ' spider' ' _______' ' __________' | 4 |
| 27 | ' spiders' ' ________' ' spider' ' ____' ' insects' ' six' | 3 |
| 28 | ' spiders' ' spider' ' ________' ' ____' ' six' ' eight' | 2 |
| 29 | ' six' ' spiders' ' eight' ' ________' ' four' ' spider' | 6 |
| 30 | ' six' ' ________' ' ____' ' eight' ' spiders' ' four' | 12 |
| 31 | ' six' ' spiders' ' eight' ' spider' ' four' ' called' | 4 |
| 32 | ' six' ' eight' ' called' ' four' ' ' ' five' | 27 |

**Best 'spider' rank: 2 at layer 28** (rank 1 = surfaced as top token in the lens)
→ 'spider' rises into the lens top-5 in the mid network: weak/partial phenomenon.
