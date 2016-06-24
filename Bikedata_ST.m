function Bikedata(filename) 

close all;
 
Fs = 10; % Visually inspecting the oscilloscope
%filename = 'Bike_1.csv'
SENS_DATA = importdata(filename);
parsedata = SENS_DATA.data;
dlen = length(parsedata(:,1));
fdel = 1;
edel = 1;
SENS_DATA = parsedata(fdel*Fs:dlen-edel*Fs,:);
% SENS_DATA = parsedata(1:timedel*Fs,:);


Accel_X = SENS_DATA(:,1);
Accel_Y = SENS_DATA(:,2);
Accel_Z = SENS_DATA(:,3);

% Remove DC Power

accel = [Accel_X Accel_Y Accel_Z];


% Time length of signal
sigtime = length(Accel_X)/Fs


powX = bandpower(Accel_X); % Find the power contained between 30 and 40s of Acceleration data after removing 20s noise.
powY = bandpower(Accel_Y);
powZ = bandpower(Accel_Z);
powers = [powX powY powZ];
ptot = sum(powers);
[~,powidx] = max(powers);

npowX = bandpower(Accel_X(1:11*Fs));
npowY = bandpower(Accel_Y(1:11*Fs));
npowZ = bandpower(Accel_Z(1:11*Fs));

% % Remove the signal with the least power.
% % This may be redundant considering weighting.
% 
% switch powidx
%     case 1
%         
%         Accel_X(:) = 0;
%         disp('X has lowest power')
%     case 2
%         Accel_Y(:) = 0;
%         powY = 0;
%         disp('Y has lowest power')
%     case 3
%         Accel_Z(:) = 0;
%         disp('Z has lowest power')
% end
%     

switch powidx % for using with max(powers)
    case 1
        data = Accel_X;
    case 2
        data = Accel_Y;
    case 3
        data = Accel_Z;
end




% Columns of correspond to following
% X 
% Y
% Z




% 
% Value1 = Accel_X(1)
% Accel_X = double(bitcmp(uint16(Accel_X)+1,'uint16'))/8192.0;
% Accel_Y = double(bitcmp(uint16(Accel_Y)+1,'uint16'))/8192.0;
% Accel_Z = double(bitcmp(uint16(Accel_Z)+1,'uint16'))/8192.0;



X_weight = 3*powX/ptot;
Y_weight = 3*powY/ptot;
Z_weight = 3*powZ/ptot;

t = linspace(0,length(Accel_X)/Fs,length(Accel_X));
length(t);



figure;
subplot(211);
plot(t,Accel_X,'b');
hold on;
plot(t,Accel_Y,'m');
plot(t,Accel_Z,'r');

xlabel('Time (sec)');
ylabel('User Acceleration (m/s^2)');
title('Cadence of Bike User');


%data = sqrt((Z_weight.*Accel_Z).^2+(Y_weight.*Accel_Y).^2+(X_weight.*Accel_X).^2);
%data = Accel_X;
data = data-mean(data);% subtract DC value
plot(t,data,'g','linewidth',1); 
hold off;
legend('X','Y','Z','Selected');


% Unfiltered Data
fftlength = 2^nextpow2(length(data));
L = length(data);
fdata = fft(data, fftlength) / L;
ctr = (fftlength / 2) + 1;
faxis = 60*(Fs / 2) .* linspace(0,1, ctr); % multiply by 60 for RPM vs RPS
mag = abs(fdata(1:ctr));
[~,idx] = max(mag);
%fftcdnc = faxis(idx)

% Plot Unfiltered Data
subplot(212);
plot(faxis,mag); hold on;
title('FFT of Weighted Averaged Data');
xlabel('Frequency (RPM)');
ylabel('Magnitude');


% Filtered Data
cutfreq = [.25 4.99];
[b,a] = butter(4,cutfreq./(Fs./2));
lpf = filter(b,a,data);
bdata = fft(lpf,fftlength) / L;
[~,idx] = max(abs(bdata(1:ctr)));
% bdata(idx) = 0;
fftcadence = faxis(idx)

plot(faxis,abs(bdata(1:ctr)),'linewidth',1);
legend('unfilt','butter LPF'); hold off;





% windlen = floor(length(lpf)/10);
windt = 4;
windlen = floor(windt*Fs);
nlap= [];
nfft=2^nextpow2(windlen+1000);
wind = hamming(windlen);
figure;
subplot 121;
spectrogram(lpf,wind,nlap,nfft,Fs,'yaxis');hold on;
% disp({'Data Length',length(lpf)})
% colorbar;
% view(2);
[s,f,t,pxx]=spectrogram(lpf,wind,nlap,nfft,Fs,'yaxis');
% whos s
title('Spectrogram');
[~,I] = max(pxx); % largest PSD in each column (STFT).
cadot = 60*f(I); %cadence over time

cadlen = length(cadot);
% cadot = [cadot;cadot(cadlen)];

subplot 122;
x = linspace(0,sigtime,cadlen)/60;
stairs(x,cadot,'linewidth',2);
grid on;
title('Discrete Cadence Over Time');
xlabel('Time (mins)');
ylabel('Cadence (RPM)');
%plot(t,f(I),'r','linewidth',1); hold off;
% avgcadence = mean(cadot)

% save output over time to .csv file
fname = strsplit(filename,'.csv');
newfilename = sprintf('%speaks.csv',fname{1});
q = [x;cadot'];
csvwrite(newfilename,q);


end
