%% backpropagation %%
clear;close all;clc;
%% data %%
w1 = [1.58,2.32,-5.8;0.67,1.58,-4.78;
    1.04,1.01,-3.63;-1.49,2.18,-0.39;
    -0.41,1.21,-4.73;1.39,3.16,2.87;
    1.20,1.40,-1.89;-0.92,1.44,-3.22;
    0.45,1.33,-4.38;-0.76,0.84,-1.96];

w2 = [0.21,0.03,-2.21;0.37,0.28,-1.8;
    0.18,1.22,0.16;-0.24,0.93,-1.01;
    -1.18,0.39,-0.39;0.74,0.96,-1.16;
    -0.38,1.94,-0.48;0.02,0.72,-0.17;
    0.44,1.31,-0.14;0.46,1.49,0.68];
w3 = [-1.54,1.17,0.64;5.41,3.45,-1.33;
    1.55,0.99,2.69;1.86,3.19,1.51;
    1.68,1.79,-0.87;3.51,-0.22,-1.39;
    1.40,-0.44,0.92;0.44,0.83,1.97;
    0.25,0.68,-0.99;-0.66,-0.45,0.08];
%% main %%
%% 3-3-1 Network %%
% initialization
ita = 0.1; % learning rate
data_w = [w1;w2]'; % training data
t1 = 1;% teacher signal:w1--t1,w2--t2
t2 = 0;
T = [repmat(t1,1,size(w1,1)),repmat(t2,1,size(w2,1))];
num = 100; % training numbers
figure;
for selection_rand = [1,0]
    if selection_rand
        % random initialization
        Wxy = rand(3,3)*2-1; 
        Wyz = rand(3,1)*2-1;
        Wyb = rand(3,1)*2-1;
        Wzb = rand(1,1)*2-1;
    else
        % fixed initialization
        Wxy = ones(3,3)*0.5; % weight from input layer to hidden layer
        Wyz = ones(3,1)*0.5; % weight from hidden layer to output layer
        Wyb = ones(3,1)*0.5; % bias of hidden layer
        Wzb = ones(1,1)*0.5; % bias of output layer
    end
    Jw =[];
    for i=1:num
        for j=1:size(data_w,2)
            [Wxy,Wyz,Wyb,Wzb] = BP_trainer(ita,Wxy,Wyz,Wyb,Wzb,data_w(:,j),T(:,j));
        end
        [Y,Z] = network(Wxy,Wyz,Wyb,Wzb,data_w);
        Jw = [Jw,sum(sum((T-Z).^2))];% Jw is the training error
    end
    if selection_rand %draw
        plot(Jw,'-');
    else
        plot(Jw,'--');
    end
    hold on;title('3-3-1 BP');legend('random initialization','fixed initialization');
end
%% 3-4-3 Network %%
% initialization
ita = 0.1; % learning rate
data_w = [w1;w2;w3]'; % training data
t1 = [1,0,0]';% teacher signal:w1--t1,w2--t2,w3--t3
t2 = [0,1,0]';
t3 = [0,0,1]';
T = [repmat(t1,1,size(w1,1)),repmat(t2,1,size(w2,1)),repmat(t3,1,size(w3,1))];
num = 100; % training numbers
figure;
for selection_rand = [1,0]
    if selection_rand
        % random initialization
        Wxy = rand(3,4)*2-1; 
        Wyz = rand(4,3)*2-1;
        Wyb = rand(4,1)*2-1;
        Wzb = rand(3,1)*2-1;
    else
        % fixed initialization
        Wxy = ones(3,4)*0.5; % weight from input layer to hidden layer
        Wyz = ones(4,3)*0.5; % weight from hidden layer to output layer
        Wyb = ones(4,1)*0.5; % bias of hidden layer
        Wzb = ones(3,1)*0.5; % bias of output layer
    end
    Jw =[];
    for i=1:num
        for j=1:size(data_w,2)
            [Wxy,Wyz,Wyb,Wzb] = BP_trainer(ita,Wxy,Wyz,Wyb,Wzb,data_w(:,j),T(:,j));
        end
        [Y,Z] = network(Wxy,Wyz,Wyb,Wzb,data_w);
        Jw = [Jw,sum(sum((T-Z).^2))];% Jw is the training error
    end
    if selection_rand %draw
        plot(Jw,'-');
    else
        plot(Jw,'--');
    end
    hold on;title('3-4-3 BP');legend('random initialization','fixed initialization');
end
function [Wxy,Wyz,Wyb,Wzb] = BP_trainer(ita,Wxy,Wyz,Wyb,Wzb,X,T)
%% BP trainer %%
% ita is the learning rate
% X is the traning data
% T is the teacher signal
% Wxy is the weight from input layer to hidden layer
% Wyz is the weight from input layer to hidden layer

[Y,Z] = network(Wxy,Wyz,Wyb,Wzb,X);

Ez = (T-Z).*Z.*(1-Z); % output layer backpropagation error
d_Wyz =  ita * Ez * Y';

Ey = Wyz * Ez.*Y.*(1-Y); % hidden layer backpropagation error
d_Wxy =  ita * Ey * X';

Wxy = Wxy + d_Wxy'; % update Wxy
Wyz = Wyz + d_Wyz'; % update Wyz

Wyb = Wyb + ita * Ey; % update Wyb
Wzb = Wzb + ita * Ez; % update Wzb
end

function [Y,Z] = network(Wxy,Wyz,Wyb,Wzb,X)
%% This function will calculate each layer value 
% X,Y,Z:each colum is a sample
% X is the data
% Y is the hidden layer result
% Z is the output layer result

X_norm = [ones(1,size(X,2));X];% normlization
Wxy = [Wyb';Wxy];% normlization
Wyz = [Wzb';Wyz];% normlization

Y = Wxy' * X_norm;
Y = 1./(1+exp(-Y));% activation function

Y_norm = [ones(1,size(Y,2));Y];% normlization
Z = Wyz' * Y_norm;
Z = 1./(1+exp(-Z));% activation function
end