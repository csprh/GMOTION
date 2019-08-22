%%fit a sinusoid to noisy data
%
% y = A sin (x + p)
%
% test script to solve for A and p with a linear inversion given x and y
%
% TJW 4 May 2017

clear
close all

%%1. create synthetic data

A = 1 % amplitude
p = pi/4 % phase / radians

x = [0:6:121*6]/365;  % time in years.. 2 years

y_clean = A * sin(2*pi*x + p); %perfect sinusoid. note the conversion of x to radians
y_noisy = y_clean + 0.5*randn(1,length(x)); % add noise with amplitude 0.5 (feel free to adjust)

%plot simulated data
plot(x,y_clean,'go')
hold on
plot(x,y_noisy,'g+')


%%2 Set up linear inversion, using trig identities
%
% y = A cos(p)sin(x) + A sin(p)cos(x)
%
% [sin(x) cos(x)]*[A cos(p);A sin(p)] = y
% G                m                  = b

G = [sin(2*pi*x') cos(2*pi*x')] ; % note "'" to make them columns
b = y_noisy';

m = G\b %quick matlab way of solving this linear inverse problem

%explicit way of doing the same thing
% G'G m = G' b (where G' is transpose of G)
% m = inv(G'G)G'b
m2 = inv(G'*G)*G'*b
% you can verify that m and m2 are identical

%% 3. Find A and p from m

p_solved=atan2(m(2),m(1)) % use atan2
A_solved = m(1)/cos(p_solved)

%confirmed works
%
%plot model predictions using best fit values of A and p 
y_model = A_solved * sin(2*pi*x + p_solved);
plot(x,y_model,'b-')
