function [ccoef1, ccoef2] = fitSinusoidMetric(y_noisy)
l = length(y_noisy);
x = [0:6:(l-1)*6]/365.25;

G = [sin(2*pi*x') cos(2*pi*x')] ; % note "'" to make them columns
b = y_noisy';

m = G\b;  %quick matlab way of solving this linear inverse problem

p_solved=atan2(m(2),m(1));
A_solved = m(1)/cos(p_solved);

%plot model predictions using best fit values of A and p 
y_model = A_solved * sin(2*pi*x + p_solved);
ccoef1 = sum(y_model.*y_noisy)/sqrt(sum(y_model.*y_model)*sum(y_noisy.*y_noisy));
%ccoef2 = xcorr(y_model,y_noisy,'normalized');
ccoef2 = crosscorr(y_model,y_noisy,1);
ccoef2 = ccoef2(2);
%close all;
%plot(y_model,'r');hold on;
%plot(y_noisy,'k');

