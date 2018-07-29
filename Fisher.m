%%%% Proj03-02: FDA %%%%%%
clear;close all;clc;
%%data sample
w1=[0.42 -0.087 0.58 ;-0.2 -3.3 -3.4 ;1.3 -0.32 1.7 ;0.39 0.71 0.23 ;-1.6 -5.3 -0.15 ;
    -0.029 0.89 -4.7;-0.23 1.9 2.2 ;0.27 -0.3 -0.87 ;-1.9 0.76 -2.1 ;0.87 -1.0 -2.6];
w2=[-0.4 0.58 0.089; -0.31 0.27 -0.04;0.38 0.055 -0.035 ;-0.15 0.53 0.011 ;-0.35 0.47 0.034 ;
    0.17 0.69 0.1;-0.011 0.55 -0.18 ;-0.27 0.61 0.12 ;-0.065 0.49 0.0012 ;-0.12 0.054 -0.063];
w3=[0.83 1.6 -0.014;1.1 1.6 0.48 ;-0.44 -0.41 0.32 ;0.047 -0.45 1.4 ;0.28 0.35 3.1 ;
    -0.39 -0.48 0.11;0.34 -0.079 0.14 ;-0.3 -0.22 2.2 ;1.1 1.2 -0.46 ;0.18 -0.11 -0.49];
w = FDA_w(w2,w3); %计算最优方向矢量
v = [1;2;-1.5];
w = normc(v);
y2 = w'*w2';% y2为w2在方向矢量w上的标量
y3 = w'*w3';
[min_value,min_ind]=min([y2 y3]);
[max_value,max_ind]=max([y2 y3]);

loc2 = w*y2; % y2在直线的坐标
loc3 = w*y3;
loc=[loc2,loc3];
figure;
l1=plot3(w2(:,1),w2(:,2),w2(:,3),'o');
hold on; l2=plot3(w3(:,1),w3(:,2),w3(:,3),'+');
plot3([loc(1,min_ind),loc(1,max_ind)],[loc(2,min_ind),loc(2,max_ind)],[loc(3,min_ind),loc(3,max_ind)]);%画直线
l3=plot3(loc2(1,:),loc2(2,:),loc2(3,:),'g*');
l4=plot3(loc3(1,:),loc3(2,:),loc3(3,:),'r.');
title('FDA'); 
legend([l1 l2 l3 l4],'pattern 2','pattern 3','pattern 2 on w','pattern 3 on w');
title('在非最优分类空间投影结果')
%%%% 设计一维贝叶斯分类器 %%%%
m_y2 = mean(y2); m_y3 = mean(y3);
S_y2 = cov(y2); S_y3 = cov(y3);

%Bayes classifier
y=[y2,y3];
likelihood2 = normpdf(y,m_y2, sqrt(S_y2)); %likelihood
likelihood3 = normpdf(y,m_y3, sqrt(S_y3)); 
result = likelihood2 - likelihood3; %因为先验概率相同，所以通过直接比较似然概率大小进行分类
result(find(result>=0))=2;%pattern 2
result(find(result<0))=3;%pattern 3
result_correct=[2*ones(1,size(y2,2)),3*ones(1,size(y3,2))];
num=size(find(result_correct-result~=0),2);
fprintf('错分点的个数：%d\n',num);


function  w  = FDA_w( w1, w2 )
%   Find FDA optimal direction Vector
%   w is optimal direction Vector
%   w1,w2 are two data sample
m1 = mean(w1)'; m2 = mean(w2)';
N1=size(w1,1);
N2=size(w2,1);
mm1 = repmat(m1,1,N1);
mm2 = repmat(m2,1,N2);
S1=(w1'-mm1)*(w1'-mm1)';
S2=(w2'-mm2)*(w2'-mm2)';
Sw=S1+S2;
w = inv(Sw)*(m1-m2);
w = normc(w);%normalization
end
