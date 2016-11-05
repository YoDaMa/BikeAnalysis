close all;
% This function is designed to test the capabilities of the power fraction
% calculation. Given a known 'max' index we seek to compute the percentage
% of the signal that is contained within the max power. 
% As a baseline, if the max index corresponds to say '60RPM', then for a
% sinusoid input of omega = 2*pi*60/60, the percentage calculated should be
% 100%. This will allow for better relative calculations in the future.

midx = 60; % RPM Value of the max...
Fs = 100; % Sampling frequency
T = 1/Fs; % The sampling period
L = 1000; % length of the signal. 
t = (0:L-1)*T; % Time axis. Same length as L, scaled by sampling period.
y = sin(2*pi*midx/60.*t); % Time domain signal
figure;
plot(t,y);
title('Time Series X(t)');

% Filter the data using a butterworth filter.
cutfreq = [.2 5]./(Fs/2); % Normalized frequency (pi rad/sample)
[b,a] = butter(4,cutfreq);
yfilt = filter(b,a,y);

% Map to frequency domain.
fftlength = 2^(nextpow2(length(y))+3); % Length of FFT.
Y = fft(y,fftlength); % FFT with zero padding
P2 = abs(Y/length(y)); % Normalize by length of original signal
P1 = P2(1:length(Y)/2+1); % grab positive frequency
P1(2:end-1) = 2*P1(2:end-1); % Multiply by 2 for same total as original
[M,idx] = max(P1);
disp('Total Power:');
pTotal = bandpower(yfilt) % Total power in signal.
% NOTE: This method for computing the power in the band needs to be adjusted
% to instead calculate the bandpower contained within a sinusoid of the same
% frequency.
disp('Power in the Max');
pBand = norm(M,2).^2 % Power in just the max of the spectrum.
f = 60*(0:Fs/(fftlength):Fs/2); % RPM frequency axis for positive freqs
figure;
plot(f,P1);
title('Single-Sided Amplitude Spectrum of X(t)');
% 
% t = linspace(0,length(Accel_X)/Fs,length(Accel_X));
% hold on;
% xlabel('Time (sec)', 'FontSize' ,10);
% ylabel('User Acceleration (m/s^2)', 'FontSize', 10);
% title('Cadence of Bike User','FontSize',10);
% data = Accel_X;
% data = data-mean(data);% subtract DC value
% plot(t,data,'linewidth',2); 
% % findpeaks(data, Fs,'MinPeakDistance',1/100,'MinPeakHeight',0);
% 
% % legend('X','Y','Z','Selected');
% 
% fftlength = 2^(nextpow2(length(data))+3);
% L = length(data);
% ctr = (fftlength / 2) + 1; 
% faxis = 60*(Fs / 2) .* linspace(0,1, ctr);    
% cutfreq = [.25 15];
% [b,a] = butter(4,cutfreq./(Fs./2));
% lpf = filter(b,a,data);
% bdata = fft(lpf,fftlength) / L;
% [M,idx] = max(abs(bdata(1:ctr)));
% 
% % bdata(idx) = 0;
% disp('Cadence:');
% fftcadence = faxis(idx)
% cadot = [cadot,fftcadence];
% disp('Powers:');
% powah = bandpower(lpf,Fs,[fftcadence/60-.001 fftcadence/60+.001])
% powah = norm(M,2)^2
% totpowah = bandpower(lpf)
% rate = powah/totpowah
% powpow = [powpow,totpowah];
% rateplot = [rateplot,rate];