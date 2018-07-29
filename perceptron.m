%% Proj05-01:perceptron %%
clear;close all;clc;
%% data %%
w1 = [0.1 1.1; 6.8 7.1; -3.5 -4.1; 2.0 2.7; 4.1 2.8; 3.1 5.0; -0.8 -1.3; 0.9 1.2; 5.0 6.4; 3.9 4.0];
w2 = [7.1 4.2; -1.4 -4.3; 4.5 0.0; 6.3 1.6; 4.2 1.9; 1.4 -3.2; 2.4 -4.0; 2.5 -6.1; 8.4 3.7; 4.1 -2.2];
w3 = [-3.0 -2.9; 0.54 8.7; 2.9 2.1; -0.1 5.2; -4.0 2.2; -1.3 3.7; -3.4 6.2; -4.1 3.4; -5.1 1.6; 1.9 5.1];
w4 = [-2.0 -8.4; -8.9 0.2; -4.2 -7.7; -8.5 -3.2; -6.7 -4.0; -0.5 -9.2; -5.3 -6.7; -8.7 -6.4; -7.1 -9.7; -8.0 -6.3];

%% initialize data
theta = 0.5;% threshold
a = [0,0,0]'; % initialize a = [w0,w]';
Y_w1 = [ones(size(w1,1),1),w1];
Y_w2 = [ones(size(w2,1),1),w2];
Y_w3 = [ones(size(w3,1),1),w3];
Y_w4 = [ones(size(w4,1),1),w4];

%% w1 and w2 %%
Y = [Y_w1;-Y_w2]';% normalization
figure;title('Jp(a) function of w1 and w2');hold on;
ita = [0.1,0.2,0.4];% learning rate
for i = 1:length(ita)
    [ak,Jpa,k]  = perceptron_trainer1( Y, ita(i), a, theta );
    if ita(i) ==ita(1),plot(0:k-1,Jpa,'--');end
    if ita(i) ==ita(2),plot(0:k-1,Jpa,'-o');end
    if ita(i) ==ita(3),plot(0:k-1,Jpa,'-*');end
end
legend('\eta = 0.1','\eta = 0.2','\eta = 0.4');
%% w2 and w3 %%
Y = [Y_w2;-Y_w3]';% normalization
figure;title('Jp(a) function of w2 and w3');hold on;
ita = [0.4,0.8,1.0];
for i = 1:length(ita)
    [ak,Jpa,k]  = perceptron_trainer1( Y, ita(i), a, theta );
    if ita(i) ==ita(1),plot(0:k-1,Jpa,'--');end
    if ita(i) ==ita(2),plot(0:k-1,Jpa,'-o');end
    if ita(i) ==ita(3),plot(0:k-1,Jpa,'-*');end
end
legend('\eta = 0.4','\eta = 0.8','\eta = 1.0');
%% w1 and w3 %%
ita = [0.01,0.02]; % learning rate
Y = [Y_w1;-Y_w3]';% normalization
b = [0.1,0.5];% margin
figure;title('Jr(a) function of w1 and w3');hold on;
for  i=1:length(b)
    for j = 1:length(ita)
        [ak,Jra,k]  = perceptron_trainer2( Y, ita(j), a, b(i) );
        if b(i) ==b(1)&&ita(j) ==ita(1),plot(0:k-2,Jra,'-');end
        if b(i) ==b(1)&&ita(j) ==ita(2),plot(0:k-2,Jra,'--');end
        if b(i) ==b(2)&&ita(j) ==ita(1),plot(0:k-2,Jra,'o');end
        if b(i) ==b(2)&&ita(j) ==ita(2),plot(0:k-2,Jra,'.');end
    end

end
legend('\eta = 0.01,b = 0.1','\eta = 0.01,b = 0.5','\eta = 0.02,b = 0.1','\eta = 0.02,b = 0.5');


function  [a,Jpa,k]  = perceptron_trainer1( Y, ita, a, theta )
%% perceptron trainer function %%
%%% input %%%
% Y is the set of after normalized data
% ita is the learning rate
% a is the initial weight vector
% theta is the threshold
%%% output %%%
% a is the solution vector
% Jpa is the criterion function
% k is the number of steps
k=0;
Jpa=[];
while(1)
    k = k + 1;
    Yk = Y(:,a'*Y<=0);% error samples
    Jpa = [Jpa,sum(-a'*Yk,2)];
    dJpa = sum(Yk,2);
    a = a + ita * dJpa; 
    if norm(ita*dJpa)<theta
        break;
    end
end
end

function  [a,Jra,k]  = perceptron_trainer2( Y, ita, a, b )
%% perceptron trainer function with relaxation procedures %%
%%% input %%%
% Y is the set of after normalized data
% ita is the learning rate
% a is the initial weight vector
% theta is the threshold
% b is the margin
%%% output %%%
% a is the solution vector
% Jra is the criterion function
% k is the number of steps
k=0;
Jra=[];
while(1)
    k = k + 1;
    Yk = Y(:,a'*Y<=b);
    if isempty(Yk)|| k>1000 % no more than 1000 times
        break;
    end
    mo = [];
    dJra = [0;0;0];
    Jra_i = 0;
    for i = 1:size(Yk,2)
        mo_2 = Yk(:,i)'*Yk(:,i);
        dJra = dJra + (b-a'*Yk(:,i))/mo_2.*Yk(:,i);
        Jra_i = Jra_i + (a'*Yk(:,i)-b)*(a'*Yk(:,i)-b)/mo_2;
    end
    Jra = [Jra,0.5*Jra_i];
    a = a + ita .* dJra;
end
end
