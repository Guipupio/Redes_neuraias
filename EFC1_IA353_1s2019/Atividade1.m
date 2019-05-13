clear all
% Carrega arquivos de Dados
file = load("/home/pupio/Documents/ia353/EFC1_IA353_1s2019/data.mat");

% Obtem matrizes de dados para treino
data_treino = file.X(1:40000,:);


% Adicionando coluna de 1
data_treino = [ones(size(data_treino, 1), 1) data_treino];

% Obtem matrizes de Resultado esperado 
esperado_treino = file.S(1:40000,:);
%esperado_treino(25001:40000,:) = file.S(45001:60000,:);


disp("Carregamento 1 OK")

% Carrega arquivos de Teste
%file = load("/home/pupio/Documents/ia353/EFC1_IA353_1s2019/test.mat");

% Obtem matrizes de dados para teste
%data_teste = file.Xt;
data_teste = file.X(40001:60000,:);

% Adicionando coluna de 1
data_teste = [ones(size(data_teste, 1), 1) data_teste];

% Obtem matrizes de Resultado esperado 
%esperado_teste = file.St;
esperado_teste = file.S(40001:60000,:);

disp("Carregamento 2 OK")

data_treino_transposta = data_treino';
a = data_treino_transposta*data_treino;
b = data_treino_transposta*esperado_treino;

desempenho = zeros(length(-10:1:10), 1);
dimensao = length(data_treino(1,:));
ind = 1;

disp("Inicio Tratamento desempenho")

for i = -8:1:12
    inv_a = (a + (2^i) * eye(dimensao))^-1;
    W = inv_a*b;
    saida_teste = data_teste*W;
    
    % Taxa de acerto
    desempenho(ind) = calc_perc(saida_teste, esperado_teste);
    
    % MEAN SQUARE ERROR
    %aux = abs(saida_teste-esperado_teste).^2;
    %MSE(ind) = sum(aux(:))/numel(saida_teste);
    aux = abs(saida_teste-esperado_teste);
    MSE(ind) = dot(aux(:),aux(:))/numel(saida_teste);
    
    % Cij
    
    X(ind) = 2^i;
    ind += 1;
endfor

% Grafico Desempenho X coeficiente:
figure(11)
semilogx(X,desempenho)
title("Taxa de classificacao X Coeficiente de Regularizacao")
xlabel("Coeficiente de Regularizacao")
ylabel("Taxa de classificacao correta")
grid;

figure(12);
semilogx(X,MSE)
title("Erro Quadratico medio X Coeficiente de Regularizacao")
xlabel("Coeficiente de Regularizacao")
ylabel("Erro Quadratico Medio")
grid;

%% Melhor Resposta: lambda = 64
i = 6
inv_a = (a + (2^i) * eye(dimensao))^-1;
W = inv_a*b;

%HeatMap(W)

