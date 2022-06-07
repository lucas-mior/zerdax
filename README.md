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
    * 1 bispo de casas claras
    * 1 bispo de casas escuras
    * 3 cavalos
    * 8 peões
- O usuário possívelmente terá de efetuar pequenos ajustes na posição
encontrada, uma vez que o algoritmo não é perfeito. 
Pode-se usar o lichess: https://lichess.org/editor

## Descrição do algoritmo
Converter uma posição de xadrez num tabuleiro real para a notação FEN consiste nas seguintes etapas:

1. Aquisição da foto (diagonal em relação ao tabuleiro).
2. Tratamento de qualidade da foto (equalização de histograma).
3. Converter foto para escala de cinza.
4. Detectar linhas retas na foto e determinar região provável que se encontra o tabuleiro.
5. Cortar foto de acordo com o passo 3, considerando uma margem adicionada para não perder nenhuma informação.
6. Reposicionar foto cortada para uma posição padrão (casa branca no canto inferior direito).
7. Aplicar algoritmo morfológico de detecão de linhas retas menores (casas) para distinguir das peças (linhas curvas).
8. Remover cor das peças para considerar apenas o formato das mesmas.
9. Utilizar o algoritmo de machine learning para determinar a probabilidade de cada peça.
10. Aplicar threshold na foto do passo 6 para determinar cores das casas e das peças.
11. A partir dos resultados do passo 9 e 10, eliminar possibilidades restritas (ver [Limitações](##Limitações))
12. Juntar as informações do passo 10 e 11 para determinar a posição completa do tabuleiro.
13. Converter informação do programa para a notação FEN.
