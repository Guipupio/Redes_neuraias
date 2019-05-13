clear all
% Gerando pesos aleatorios com desvio padrao de 0.2
pesos_aleatorios = 0.2 .* randn(785,500);

% Carrega arquivos de Dados
file = load("/home/pupio/Documents/ia353/EFC1_IA353_1s2019/data.mat");

% Obtem matrizes de dados para treino
data_treino = file.X(1:40000,:);

% Adicionando coluna de 1
data_treino = [ones(size(data_treino, 1), 1) data_treino];
data_treino = tanh(data_treino*pesos_aleatorios);

% Obtem matrizes de Resultado esperado 
esperado_treino = file.S(1:40000,:);

% Obtem matrizes de dados para teste
data_teste = file.X(40001:60000,:);

% Adicionando coluna de 1
data_teste = [ones(size(data_teste, 1), 1) data_teste];
data_teste = tanh(data_teste*pesos_aleatorios);

% Obtem matrizes de Resultado esperado 
esperado_teste = file.S(40001:60000,:);

disp("Carregamento 2 OK")

data_treino_transposta = data_treino';
a = data_treino_transposta*data_treino;
b = data_treino_transposta*esperado_treino;

desempenho = zeros(length(-10:1:10), 1);
dimensao = length(data_treino(1,:));
ind = 1;

disp("Inicio Tratamento desempenho")

for i = -16:1:9
    inv_a = (a + (2^i) * eye(dimensao))^-1;
    W = inv_a*b;
    saida_teste = data_teste*W;
    
    % Taxa de acerto
    desempenho(ind) = calc_perc(saida_teste, esperado_teste);
    
    
    % MEAN SQUARE ERROR
    aux = abs(saida_teste-esperado_teste).^2;
    MSE(ind) = sum(aux(:))/numel(saida_teste);
    
    % Cij
    
    X(ind) = 2^i;
    ind += 1;
endfor

% Grafico Desempenho X coeficiente:
figure(3)
semilogx(X,desempenho)
title("Coeficiente de Regularizacao X Taxa de classificacao - ATV2")
xlabel("Coeficiente de Regularizacao")
ylabel("Taxa de classificacao correta")
grid;

figure(4);
semilogx(X,MSE)
title("Coeficiente de Regularizacao X Erro Quadratico medio - ATV2")
xlabel("Coeficiente de Regularizacao")
ylabel("Erro Quadratico Medio")
grid;

