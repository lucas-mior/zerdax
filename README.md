# zerdax

Zerdax é um programa capaz de gerar a notação FEN de uma posição
de xadrez a partir de sua foto.

## Limitações
Atualmente, algumas restrições são feitas para garantir o funcionamento:
- Somente fotos de um tabuleiro específico
- Posição da fotografia deve ser diagonal em relação ao tabuleiro
- Assume-se que em uma dada posição, um jogador terá no máximo:
    * 1 rei
    * 2 damas
    * 2 torres
    * 2 bispos
    * 3 cavalos
    * 8 peões
- O usuário possívelmente terá de efetuar pequenos ajustes na posição
encontrada, uma vez que o algoritmo não é perfeito. 
Pode-se usar o lichess: https://lichess.org/editor

