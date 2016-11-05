
close all;
midx = 64.4531/60;
Fs = 100;
T = 1/Fs;
L = 1000;
t = (0:L-1)*T;
disp('Length');
length(1:L/2+1)
length(t);
y = sin(2*pi*midx.*t) ;
figure;
plot(t,y)
fftlength = 2^12;
Y = fft(y,fftlength);
figure;
plot(abs(Y));
P2 = abs(Y/length(y)).^2;
length(Y)
P1 = P2(1:length(Y)/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = 60*(0:Fs/(fftlength):Fs/2);
f2 = 60*(0:Fs/(fftlength):Fs-Fs/fftlength);
length(P2)
length(f2)


figure;
plot(f,P1);
figure;
plot(f2,P2);
title('Single-Sided Amplitude Spectrum of X(t)');
% plot(abs(f/L));