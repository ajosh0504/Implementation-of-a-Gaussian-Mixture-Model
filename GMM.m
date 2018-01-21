%SOURCE:  https://www.mathworks.com/matlabcentral/fileexchange/45817-expectation-maximization-algorithm-with-gaussian-mixture-model
clc;
clear all;
close all;
data=load('gmm.txt');
[m,n]=size(data);
data=reshape(data,[1,m]);

% Initial guess (random guess)
temp=randperm(length(data));
piecap(1)=0.5;
meucap1(1)=data(temp(1));
meucap2(1)=data(temp(2));
sigmacap1(1)=var(data);
sigmacap2(1)=var(data);

% Plotting initial basis
x = [-30:0.1:30];
y1 = gauss_dist(x, meucap1(1), sigmacap1(1));
y2 = gauss_dist(x, meucap2(1), sigmacap2(1));
figure
plot(x,y1,'b-');
hold on;
plot(x,y2,'r-');
hold on;
for i = 1:100 
% Expectation Step
Qq1=gauss_dist(data,meucap1(i),sigmacap1(i));
Qq2=gauss_dist(data,meucap2(i),sigmacap2(i));
log_likelihood(i)=sum(log(((1-piecap(i))*Qq1) + (piecap(i)*Qq2)));
responsibilities(i,:)=(piecap(i)*Qq2)./(((1-piecap(i))*Qq1)+(piecap(i)*Qq2));

% Maximization Step 
    
meucap1(i+1)=sum((1-responsibilities(i,:)).*data)/sum(1-responsibilities(i,:));
meucap2(i+1)=sum((responsibilities(i,:)).*data)/sum(responsibilities(i,:));
    
sigmacap1(i+1)=sum((1-responsibilities(i,:)).*((data-meucap1(i)).^2))/sum(1-responsibilities(i,:));
sigmacap2(i+1)=sum((responsibilities(i,:)).*((data-meucap2(i)).^2))/sum(responsibilities(i,:));
    
piecap(i+1)=sum(responsibilities(i,:))/length(data);

end
%Plotting estimated GaussiansI(in black) after running EM algorithms. 
%Randomly generated initial conditions using randperm.
y1 = gauss_dist(x, meucap1(2), sigmacap1(2));
y2 = gauss_dist(x, meucap2(2), sigmacap2(2));
plot(x,y1,'k-');
hold on;
plot(x,y2,'k-');
%Plotting max likelihood.
figure
plot(log_likelihood);
xlabel('Iteration');
ylabel('Observed Data Log-likelihood');