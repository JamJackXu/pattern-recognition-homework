clear;close all;clc;
 %data samples
w1 = [0.28 1.31 -6.2; 0.07 0.58 -0.78; 1.54 2.01 -1.63; -0.44 1.18 -4.32; -0.81 0.21 5.73;
    1.52 3.16 2.77; 2.20 2.42 -0.19; 0.91 1.94 6.21; 0.65 1.93 4.38; -0.26 0.82 -0.96]; 
w2 = [0.011 1.03 -0.21; 1.27 1.28 0.08; 0.13 3.12 0.16; -0.21 1.23 -0.11; -2.18 1.39 -0.19;
    0.34 1.96 -0.16; -1.38 0.94 0.45; -0.12 0.82 0.17; -1.44 2.31 0.14; 0.26 1.94 0.08];
w3 = [1.36 2.17 0.14; 1.41 1.45 -0.38; 1.22 0.99 0.69; 2.46 2.19 1.31; 0.68 0.79 0.87;
    2.51 3.22 1.35; 0.60 2.44 0.92; 0.64 0.13 0.97; 0.85 0.58 0.99;0.66 0.51 0.88];
x1 = [0.5 1.0 0.0]'; x2 = [0.31 1.51 -0.5]'; x3 = [-0.3 0.44 -0.1]';
%%%%%%%%%%%%%%%%%%%%%%%%  Parzen  %%%%%%%%%%%%%%%%%%%%%%%%%
for h = [0.1,1]%Parzen window width
sigma = h;
result1 = Bayes_classifier(w1,w2,w3,x1,h);
result2 = Bayes_classifier(w1,w2,w3,x2,h);
result3 = Bayes_classifier(w1,w2,w3,x3,h);
fprintf('h is %f,result of Bayes\n x1 is class:%d | x2 is class:%d | x3 is class:%d\n',h,result1,result2,result3);
[w,a] = PNN_trainer(w1,w2,w3);
result4 = PNN_classifier(w,a,sigma,x1);
result5 = PNN_classifier(w,a,sigma,x2);
result6 = PNN_classifier(w,a,sigma,x3);
fprintf('h is %f,result of PNN\n x1 is class:%d | x2 is class:%d | x3 is class:%d\n',h,result4,result5,result6);
end
%%%%%%%%%%%%%%%%%%%%%%%%  KNN  %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% 1-D data %%%%%%%%%%%%%%%%
data = w3(:,1);
for k = [3,5] % k value
    N = 100;% numbers of test data
    test_x = linspace(min(data)-1,max(data)+1,N); 
    n = size(data,1);% numbers of samples
    dist = zeros(N,n);% distance between test data and sample data
    for i = 1:N
        for j = 1:n
            dist(i,j) = sqrt((test_x(i)-data(j))^2);
        end
    end
    [a,b] = sort(dist,2);% sort each of its rows in ascending order
    v = a(:,k); % v is a line
    p = k/n./v; %probability
    if k==3 % draw picture
        plot(test_x,p,'-');
    else
        plot(test_x,p,'--');
    end
    hold on
end
legend('k=3','k=5');
xlabel('x');
ylabel('p(x)');
%%%%%%%%%%%%%%%% 2-D data %%%%%%%%%%%%%%%%
data = w3(:,1:2);
for k = [3,5] % k value
    N = 100;% numbers of test data
    test_x1 = linspace(min(data(:,1))-1,max(data(:,1))+1,N);
    test_x2 = linspace(min(data(:,2))-1,max(data(:,2))+1,N);
    [X,Y] = meshgrid(test_x1,test_x2);% generate grid
    n = size(data,1);% numbers of samples
    dist = zeros(N,n);% distance between test data and sample data
    for i = 1:size(X(:),1)
        for j = 1:n
            dist(i,j) = sqrt((X(i)-data(j,1))^2+(Y(i)-data(j,2))^2);
        end
    end
    [a,b] = sort(dist,2);% sort each of its rows in ascending order
    v = pi*a(:,k).^2;% v is a circle
    p = k/n./v; %probability
    p = reshape(p,size(X));
    if k==3 % draw picture
        figure,mesh(X,Y,p);
        title('k=3');
    else
        figure,mesh(X,Y,p);
        title('k=5');
    end
end
%%%%%%%%%%%%% knn classifier %%%%%%%%%%%%%%
x4=[-0.41,0.82,0.88]';x5=[0.14,0.72,4.1]';x6=[-0.81,0.61,-0.38]';
for k=[3,5]
    result7 = KNN_classifier(w1,w2,w3,x4,k);
    result8 = KNN_classifier(w1,w2,w3,x5,k);
    result9 = KNN_classifier(w1,w2,w3,x6,k);
    fprintf('k is %f,result of KNN\n x4 is class:%d | x5 is class:%d | x6 is class:%d\n',k,result7,result8,result9);

end

function p = parzen(w,x,h)
% This function is to calculate p(x|w) based on Parzen window
% w is samples
% h is Parzen window width
% x is test data
n = size(w,1);%numbers of samples
p = 0;
for i = 1:n
	p = p+exp(-(w(i,:)'-x)'*((w(i,:)'-x))/(2*h^2));
end
p = p/n;
end

function result = Bayes_classifier(w1,w2,w3,x,h)
% this is a Bayes classifier
% w1,w2,w3 are training data
% x is test data
% h is Parzen window width
% pw1=pw2=pw3=1/3
likelihood1 = parzen(w1,x,h);
likelihood2 = parzen(w2,x,h);
likelihood3 = parzen(w3,x,h);
[val,result] = max([likelihood1,likelihood2,likelihood3]);
end

function [w a] = PNN_trainer(w1,w2,w3)
% PNN trainer of 3 classes
% w is weight
% a is an index matrix of 3 classes
norm_w1 = normr(w1);% normalization
norm_w2 = normr(w2);
norm_w3 = normr(w3);
w = [norm_w1;norm_w2;norm_w3];
a = zeros(size(w));
for j = 1:size(w,1)
	if j <= size(w1,1)
        a(j,1) = 1;
    end
	if size(w1,1)<j && j<=size(w1,1)+size(w2,1)
        a(j,2)=1;
    end
	if size(w1,1)+size(w2,1) < j
        a(j,3) = 1;
    end
end
end

function result = PNN_classifier(w,a,sigma,x)
% PNN classifier of 3 classes
% x is test data
% w is weight
% a is an index matrix of 3 classes
% sigma os Parzen window width
x=x./norm(x);
net = w*x;
g = exp((net-1)/sigma^2);
g1 = 0;g2 = 0;g3 = 0;
for j = 1:size(a,1)
	if a(j,1) == 1
       g1 = g1+g(j);
    end
	if a(j,2) == 1
       g2 = g2+g(j);
    end
    if a(j,3) == 1
       g3 = g3+g(j);
    end
end
[val,result] = max([g1,g2,g3]);
end

function result = KNN_classifier(w1,w2,w3,x,k)
% KNN classifier of 3 classes
% w1,w2,w3 are sample data
% x is test data
w = [w1;w2;w3];
index = [ones(1,size(w1,1)),2*ones(1,size(w1,1)),3*ones(1,size(w1,1))];
n = size(w,1);% numbers of samples
dist = zeros(1,n);% distance between test data and sample data
for j = 1:n
	dist(j) = sqrt((x(1)-w(j,1))^2+(x(2)-w(j,2))^2+(x(3)-w(j,3))^2);% L-2
%     dist(j) = abs((x(1)-w(j,1)))+abs((x(2)-w(j,2)))+abs((x(3)-w(j,3)));% L-1
%     dist(j) = max((x(1)-w(j,1)))+max((x(2)-w(j,2)))+max((x(3)-w(j,3)));% L-inf
end
[a,b] = sort(dist);% sort each of its rows in ascending order
index = index(b);
index = index(1:k);
hist_index = hist(index,[1:1:k]);
[val,result] = max(hist_index);
end