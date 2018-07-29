clear,close all,clc
%pattern 1
m1=[1,3];%mean value
% S1=[1.5,0;0,1];%covariance
S1=[1.5,1;1,1];
% Pw1=0.5;%prior probability
Pw1=0.4;

%pattern 2
m2=[3,1];
% m2=[2,2];
%  m2=[4,0];
S2=[1,0.5;0.5,2];
% Pw2=0.5;
Pw2=0.6;

%data generation
n=100;%numbers
d1=mvnrnd(m1,S1,n);
d2=mvnrnd(m2,S2,n);
samples=[d1;d2];

%data visulation
figure;subplot(1,2,1)
c1=scatter(d1(:,1),d1(:,2),'.');
hold on;
c2=scatter(d2(:,1),d2(:,2),'+');
title('two patterns');
legend([c1 c2],'pattern 1','pattern 2');

%Bayes classification
likelihood1=mvnpdf(samples,m1,S1);
likelihood2=mvnpdf(samples,m2,S2);
g1=likelihood1.*Pw1;
g2=likelihood2.*Pw2;

%classification
result=zeros(size(samples,1),1);
result(find(g1-g2>0))=1;%classify samples as pattern 1
result(find(g1-g2<0))=2;%classify samples as pattern 2
result(find(g1-g2==0))=inf;%classify samples as unsure pattern

%accuracy
theoretical_result=[ones(size(d1,1),1);2*ones(size(d2,1),1)];
correct_ind=find(result-theoretical_result==0);
wrong_ind=find(abs(result-theoretical_result)==1);
unsure_ind=find(result-theoretical_result==inf);

accuracy=size(correct_ind,1)/size(samples,1);
fprintf('accuracy is %f\n',accuracy);

%classification visulation
correct=samples(correct_ind,:);
wrong=samples(wrong_ind,:);
unsure=samples(unsure_ind,:);

subplot(1,2,2)
c=scatter(correct(:,1),correct(:,2),'^');
hold on
w=scatter(wrong(:,1),wrong(:,2));
title('result of classification');
legend([c w],'correct result','wrong result');
if ~isempty(unsure_ind)
    u=scatter(unsure(:,1),unsure(:,2),'*');
    legend([c w u],'correct result','wrong result','unsure result');
end
set(gcf, 'position', [0 0 1000 400]);