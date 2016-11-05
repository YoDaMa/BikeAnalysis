function Bikedata(filename) 

close all;

Fs = 100; % Visually inspecting the oscilloscope
%filename = 'Bike_1.csv'
SENS_DATA = importdata(filename);
parsedata = SENS_DATA.data;
dlen = length(parsedata(:,1));
fdel = 20;
edel = 20;
SENS_DATA = parsedata(fdel*Fs:dlen-edel*Fs,:);
% SENS_DATA = parsedata(1:timedel*Fs,:);
timestamp = SENS_DATA(:,1);
% ATT_R = SENS_DATA(:,2);
% ATT_P = SENS_DATA(:,3);
% ATT_Y = SENS_DATA(:,4);
% ROTR_X = SENS_DATA(:,5);
% ROTR_Y = SENS_DATA(:,6);
% ROTR_Z = SENS_DATA(:,7);
% GR_X = SENS_DATA(:,8);
% GR_Y = SENS_DATA(:,9);
% GR_Z = SENS_DATA(:,10);
UA_X = SENS_DATA(:,11);
UA_Y = SENS_DATA(:,12);
UA_Z = SENS_DATA(:,13);
% MAG_X = SENS_DATA(:,12);
% MAG_Y = SENS_DATA(:,13);
% MAG_Z = SENS_DATA(:,14);


keep_rows = find(~isnan(UA_X)); % find the 1s produced by isnan.

% User acceleration
Accel_X = UA_X(keep_rows);
Accel_Y = UA_Y(keep_rows);
Accel_Z = UA_Z(keep_rows);
ts = timestamp(keep_rows);
ts = ts - ts(1);
[~,sorted] = sort(ts);
Accel_X = Accel_X(sorted);
% Accel_X(:) = 0;
Accel_Y = Accel_Y(sorted);
% Accel_Y(:) = 0;
Accel_Z = Accel_Z(sorted);
% Accel_Z(:) = 0;
% accel = [Accel_X Accel_Y Accel_Z];
ws = 40
we = 50
Accel_X = Accel_X(ws*Fs:we*Fs);
Accel_Y = Accel_Y(ws*Fs:we*Fs);
Accel_Z = Accel_Z(ws*Fs:we*Fs);

% Time length of signal

sigtime = length(Accel_X)/Fs/60.0;
writing = sprintf('\nThe signal length is %d minutes.\n',sigtime); 
disp(writing);


powX = bandpower(Accel_X); % Find the power contained between 30 and 40s of Acceleration data after removing 20s noise.
powY = bandpower(Accel_Y);
powZ = bandpower(Accel_Z);
powers = [powX powY powZ];
ptot = sum(powers);
[~,powidx] = max(powers);

% npowX = bandpower(Accel_X(1:11*Fs));
% npowY = bandpower(Accel_Y(1:11*Fs));
% npowZ = bandpower(Accel_Z(1:11*Fs));

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




X_weight = 3*powX/ptot;
Y_weight = 3*powY/ptot;
Z_weight = 3*powZ/ptot;

t = linspace(0,length(Accel_X)/Fs,length(Accel_X));




figure;
subplot(211);
% plot(t,Accel_X,'b');
hold on;
% plot(t,Accel_Y,'m');
% plot(t,Accel_Z,'r');

xlabel('Time (sec)', 'FontSize' ,10);
ylabel('User Acceleration (m/s^2)', 'FontSize', 10);
title('Cadence of Bike User','FontSize',10);


%data = sqrt((Z_weight.*Accel_Z).^2+(Y_weight.*Accel_Y).^2+(X_weight.*Accel_X).^2);
data = Accel_X;
data = data-mean(data);% subtract DC value
plot(t,data,'b','linewidth',2); 
% findpeaks(data, Fs,'MinPeakDistance',1/100,'MinPeakHeight',0);
hold off;
% legend('X','Y','Z','Selected');



% Unfiltered Data
fftlength = 2^nextpow2(length(data));
L = length(data);
fdata = fft(data, fftlength) / L;
ctr = (fftlength / 2) + 1;
faxis = 60*(Fs / 2) .* linspace(0,1, ctr); % multiply by 60 for RPM vs RPS
mag = abs(fdata(1:ctr));
disp('Maximum Index:');
[~,idx] = max(mag)
%fftcdnc = faxis(idx)

% Plot Unfiltered Data
subplot(212);
plot(faxis,mag,'LineWidth',2); hold on;
% findpeaks(mag,faxis,'MinPeakHeight',.005)
[pkt,lct] = findpeaks(mag,faxis,'MinPeakHeight',.005);
lct(1:5)
% plot(lct,pkt,'x');
[spkt,slct] = sort(pkt,'descend');
xsorted = lct(slct);
maxVal = spkt(1:5)
maxIdx = xsorted(1:5)
% plot(maxIdx,maxVal,'x');
B = mag(slct);


title('FFT of Weighted Averaged Data','FontSize' ,10);
xlabel('Frequency (RPM)','FontSize' ,10);
ylabel('Magnitude','FontSize' ,10);


% Filtered Data
cutfreq = [.25 15];
[b,a] = butter(4,cutfreq./(Fs./2));
lpf = filter(b,a,data)  ;
bdata = fft(lpf,fftlength) / L;
[~,idx] = max(abs(bdata(1:ctr)));
% bdata(idx) = 0;
fftcadence = faxis(idx)
powah = bandpower(lpf,Fs,[fftcadence/60-.05 fftcadence/60+.05])
totpowah = bandpower(lpf)

plot(faxis,abs(bdata(1:ctr)),'linewidth',1);
legend('unfilt','butter LPF'); hold off;


    
% figure;
% subplot 221;
    % windlen = floor(length(lpf)/10);
    windt = 4;
    windlen = floor(windt*Fs);
    nlap= [];
    nfft=2^nextpow2(windlen);
    wind = hamming(windlen);
%     spectrogram(lpf,wind,nlap,nfft,Fs,'yaxis');hold on;
%     title('Spectrogram (4s)');
%     % colorbar;
%     %plot(t,f(I),q,'r','linewidth',2); 
%     hold off;
%     
% 
% 
% subplot 222;
    [s,f,t,pxx]=spectrogram(lpf,wind,nlap,nfft,Fs,'yaxis');
%     disp(['Spectrogram time: ', num2str(t(length(t))),' seconds.']);
    [~,I] = max(10*log10(pxx)); % largest PSD in each column (STFT).
    cadot = 60*f(I); %cadence over time
%     cadlen = length(cadot);
%     x = linspace(0,t(length(t)),cadlen);
%     stairs(x,cadot,'linewidth',2);
%     grid on;
%     title('Discrete Cadence Over Time');
%     xlabel('Time (mins)');
%     ylabel('Cadence (RPM)');
% 
% subplot 223;
%     windt = 12.8;
%     windlen = floor(windt*Fs);
%     nlap= [];
%     nfft=2^nextpow2(windlen);
%     wind = hamming(windlen);
%     spectrogram(lpf,wind,nlap,nfft,Fs,'yaxis');hold on;
%     title('Spectrogram (12.8s)');
% 
% subplot 224;
    [~,f,t,pxx]=spectrogram(lpf,wind,nlap,nfft,Fs,'yaxis');
    [M,I] = max(10*log10(pxx)); % largest PSD in each column (STFT).
    M
    cadot = 60*f(I);
    cadlen = length(cadot);
    x = linspace(0,t(length(t)),cadlen);
%     stairs(x,cadot,'linewidth',2);
%     grid on;
%     title('Discrete Cadence Over Time');
%     xlabel('Time (mins)');
%     ylabel('Cadence (RPM)');









avgcadence = mean(cadot);

% save output over time to .csv file
fname = strsplit(filename,'.csv');
newfilename = sprintf('%speaks.csv',fname{1});
q = [x',cadot];
disp(['Window Length: ',num2str(windlen)]);
disp(['FFT Peak: ',num2str(fftcadence)]);
disp(['STFT Avg: ', num2str(avgcadence)]);
csvwrite(newfilename,q);


end
