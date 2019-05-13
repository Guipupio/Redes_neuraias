% Carrega arquivos de Dados
file = load("/home/pupio/Documents/ia353/EFC1_IA353_1s2019/data.mat");
 
% Obtem matrizes de dados para treino e teste
data_treino = file.X(1:40000, :);
data_teste = file.X(40001:60000, :);

% Obtem matrizes de Resultado esperado 
esperado_treino = file.S(1:40000, :);
esperado_teste = file.S(40001:60000, :);

% Calcula modelo linear -  Quadrados Minimos
W = data_treino\esperado_treino;

% Realiza teste com lambda = 0
saida_teste = data_teste*W;

desempenho = calc_perc(saida_teste, esperado_teste);





% Carrega arquivos de Teste
%file = load("/home/pupio/Documents/ia353/EFC1_IA353_1s2019/test.mat");

% Obtem Vetores de solucao para treino e teste
%sol_treino = file.

%resposta_esperada = file.S



%csvwrite("FILENAME", VAR)