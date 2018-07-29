%%%Project02-02%%%
clear;close all;clc;
%experiment data
w1 = [-5.01 -8.12 -3.68; -5.43 -3.48 -3.54; 1.08 -5.52 1.66; 0.86 -3.78 -4.11; -2.67 0.63 7.39;% pattern 1
    4.94 3.29 2.08; -2.51 2.09 -2.59; -2.25 -2.13 -6.94; 5.56 2.86 -2.26; 1.03 -3.33 4.33];
w2 = [-0.91 -0.18 -0.05; 1.30 -2.06 -3.53; -7.75 -4.54 -0.95; -5.47 0.50 3.92; 6.14 5.72 -4.85;% pattern 2
    3.60 1.26 4.36; 5.37 -4.63 -3.65; 7.18 1.46 -6.66; -7.39 1.17 6.30; -7.50 -6.32 -0.31];
w3 = [5.35 2.26 8.13; 5.12 3.22 -2.66; -1.34 -5.31 -9.87; 4.48 3.42 5.19; 7.11 2.39 9.21;% pattern 3
    7.17 4.33 -0.98; 5.75 3.97 6.65; 0.77 0.27 2.41; 0.90 -0.43 -8.71; 3.52 -0.36 6.43];
%�����ֵʸ����Э�������
m1 = mean(w1)'; m2 = mean(w2)'; m3 = mean(w3)';
S1 = cal_S(w1,m1); S2 = cal_S(w2,m2); S3 = cal_S(w3,m3);
%���������������Ͼ���ֱ�Ϊ1��2��3ʱ��ͼ��
subplot(2,2,1);draw(m1,S1);title('pattern 1');
subplot(2,2,2);draw(m2,S2);title('pattern 2');
subplot(2,2,3);draw(m3,S3);title('pattern 3');

%�����С���Ͼ�����������Բ��Ե���з���
test1 = [1 2 1]'; test2 = [5 3 2]'; test3 = [0 0 0]'; test4 = [1 0 0]'; test = [test1 test2 test3 test4];
D1 = mahalanobis_classifier(test1,m1,m2,m3,S1,S2,S3); %������С���Ͼ�����������з���
D2 = mahalanobis_classifier(test2,m1,m2,m3,S1,S2,S3);
D3 = mahalanobis_classifier(test3,m1,m2,m3,S1,S2,S3);
D4 = mahalanobis_classifier(test4,m1,m2,m3,S1,S2,S3);
%������Сŷʽ������������з���
E1 = oushi_classifier(test1,m1,m2,m3); 
E2 = oushi_classifier(test2,m1,m2,m3);
E3 = oushi_classifier(test3,m1,m2,m3);
E4 = oushi_classifier(test4,m1,m2,m3);
%%%���ñ�Ҷ˹���������з���
pw1 = 1/3; pw2 = 1/3; pw3 = 1/3;
result = []; %%�洢������
for i =1:1:4
    likelihood_1 = mvnpdf(test(:,i),m1,S1);%ģʽ��1����Ȼ����
    likelihood_2 = mvnpdf(test(:,i),m2,S2);%ģʽ��2����Ȼ����
    likelihood_3 = mvnpdf(test(:,i),m3,S3);%ģʽ��3����Ȼ����
    g1 = likelihood_1*pw1;
    g2 = likelihood_2*pw2;
    g3 = likelihood_3*pw3;
    [val,output]=max([g1,g2,g3]);
    result(end+1)=output;
end
function  S = cal_S( sample,m )
% �˺����Ĺ������ڼ���Э����
sample=sample';
num=size(sample,2);% numbers of samples
mm=repmat(m,1,num);
S=(sample-mm)*(sample-mm)'./num;
end

function  output  = mahalanobis_classifier( x,m1,m2,m3,S1,S2,S3 )
% �˺���Ϊ��С���Ͼ��������
    g1=-mahalanobis_distance(x,m1,S1);
    g2=-mahalanobis_distance(x,m2,S2);
    g3=-mahalanobis_distance(x,m3,S3);
    [val,output]=max([g1,g2,g3]);
end

function output = oushi_classifier( x,m1,m2,m3 )
%�˺���Ϊ��Сŷʽ���������
    E1 = -sqrt((x - m1)'*(x - m1)); 
    E2 = -sqrt((x - m2)'*(x - m2));
    E3 = -sqrt((x - m3)'*(x - m3));
    [val,output]=max([E1,E2,E3]);
end

function  g  = mahalanobis_distance( x,m,S )
% calculate the mahalanobis distance
    g = sqrt((x-m)'*inv(S)*(x-m));
end

function  draw( m,S )
%draw picture
    x=m(1)-25:1:m(1)+25; y=m(2)-25:1:m(2)+25;z=m(3)-25:1:m(3)+25;
    [X,Y,Z]=meshgrid(x,y,z);
    for D=1:3
        all=[];
        for i=1:size(X(:))
            if abs(mahalanobis_distance([X(i);Y(i);Z(i)],m,S)-D)<0.1
                all=[all,[X(i);Y(i);Z(i)]];
            end
        end
        scatter3(all(1,:),all(2,:),all(3,:));
        hold on;
    end
end