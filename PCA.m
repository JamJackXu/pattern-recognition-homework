%%% Proj03-01: PCA %%%
clear;close all;clc;
%%%%%%%%%(a)%%%%%%%%%
for N = [10,100,1000,10000,100000]
u = [5;7]; sigma = [9,2.4;2.4,1];
X = mvnrnd(u,sigma,N);
figure; subplot(2,1,1); plot(X(:,1),X(:,2),'o');title(['2D-Data X,N=',num2str(N)]);
%%%%%%%%%(b)%%%%%%%%%
m = mean(X)'; %mean value
mm=repmat(m,1,N);
S = (X'-mm)*(X'-mm)';%scatter matrix
[V,D] = eig(S); %eigenvalue and eigenvector
%%%%%%%%%(c)%%%%%%%%%
Y=V*(X'-mm);
subplot(2,1,2); plot(Y(1,:),Y(2,:),'+');title(['2D-Data Y,N=',num2str(N)]);
end
%%%%%%%%%(e)%%%%%%%%%
for N = [10,20,50,100,1000]
u = [10;15;15]; sigma = [90,2.5,1.2; 2.5,35,0.2; 1.2,0.2,0.02];
X = mvnrnd(u,sigma,N); %生成样本集合
figure; plot3(X(:,1),X(:,2),X(:,3),'o');title(['3D-Data X,N=',num2str(N)]);
%%%%%%%%%(f)%%%%%%%%%
m = mean(X)'; %mean value
mm=repmat(m,1,N);
%S = (X'-mm)*(X'-mm)';%scatter matrix
S = (N-1)*cov(X);
[V,D] = eig(S); %eigenvalue and eigenvector
[DD,indx]=sort(sum(D),'descend');%descend sort
VV=V(:,indx);
y1=VV(:,1)'*(X'-mm);
y2=VV(:,2)'*(X'-mm);
Y=[y1;y2];
figure,plot(Y(1,:),Y(2,:),'o');title(['2D-Data Y,N=',num2str(N)]);
%%%%%%%%%(g)%%%%%%%%%
VV_inv=inv(VV);
W=VV_inv(:,1:2);
Z=W*Y+mm;
figure; tuli1=plot3(X(:,1),X(:,2),X(:,3),'o');title(['3D-Data X And Y,N=',num2str(N)]);
hold on; tuli2=plot3(Z(1,:),Z(2,:),Z(3,:),'*');
legend([tuli1,tuli2],'Raw Data','PCA Data');
for i =1:1:N %%连线
    plot3([Z(1,i) X(i,1)],[Z(2,i) X(i,2)],[Z(3,i) X(i,3)]);
end
%%%%%%%%%(h)%%%%%%%%%
bias=(X'-Z).^2;
bias_sum=sum(sum(bias));
bias_average=bias_sum/N;
end





