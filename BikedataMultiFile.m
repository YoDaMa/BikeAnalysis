function BikedataMultiFile(filename) 

close all;

Fs = 100; % Visually inspecting the oscilloscope
%filename = 'Bike_1.csv'
SENS_DATA = importdata(filename);
parsedata = SENS_DATA.data;
dlen = length(parsedata(:,1));
fdel = 20;
edel = 40;
SENS_DATA = parsedata;
% SENS_DATA = parsedata(1:timedel*Fs,:);
timestamp = SENS_DATA(:,1);
UA_X = SENS_DATA(:,2);
UA_X1 = SENS_DATA(:,3);
UA_X2 = SENS_DATA(:,4);
UA_X3 = SENS_DATA(:,5);
UA_X4 = SENS_DATA(:,6);


keep_rows = find(~isnan(UA_X)); % find the 1s produced by isnan.
Accel_X = UA_X(keep_rows);

keep_rows = find(~isnan(UA_X1)); % find the 1s produced by isnan.
Accel_X1 = UA_X1(keep_rows);
keep_rows = find(~isnan(UA_X2)); % find the 1s produced by isnan.
Accel_X2 = UA_X2(keep_rows);
keep_rows = find(~isnan(UA_X3)); % find the 1s produced by isnan.
Accel_X3 = UA_X3(keep_rows);
keep_rows = find(~isnan(UA_X4)); % find the 1s produced by isnan.
Accel_X4 = UA_X4(keep_rows);
disp(size(Accel_X))
accels = [Accel_X(fdel*Fs:edel*Fs),Accel_X1(fdel*Fs:edel*Fs),Accel_X2(fdel*Fs:edel*Fs),Accel_X3(fdel*Fs:edel*Fs),Accel_X4(fdel*Fs:edel*Fs)];
rateplot = []
figure;
for i = linspace(1,5,5)
%     writing = sprintf('\nThe signal length is %d minutes.\n',sigtime); 
%     disp(writing);

    data = accels(:,i);
    t = linspace(0,length(data)/Fs,length(data));
    hold on;
    xlabel('Time (sec)', 'FontSize' ,10);
    ylabel('User Acceleration (m/s^2)', 'FontSize', 10);
    title('Cadence of Bike User','FontSize',10);
    p = bandpower(data);
    disp('Power:');
    disp(p);
    
    data = data-mean(data);% subtract DC value
    plot(t,data,'linewidth',2); 
    % findpeaks(data, Fs,'MinPeakDistance',1/100,'MinPeakHeight',0);

    % legend('X','Y','Z','Selected');
    

end
hold off;
figure; hold on;
for i = 1:5
    data = accels(:,i);
    fftlength = 2^nextpow2(length(data));
    L = length(data);
    cutfreq = [.25 15];
    [b,a] = butter(4,cutfreq./(Fs./2));
    lpf = filter(b,a,data);
    fdata = fft(lpf,fftlength) / L;
    ctr = (fftlength / 2) + 1;
    faxis = 60*(Fs / 2) .* linspace(0,1, ctr); % multiply by 60 for RPM vs RPS
    mag = abs(fdata(1:ctr));
    % disp('Maximum Index:');
    [M,idx] = max(mag);
    %fftcdnc = faxis(idx)
    ppeak = norm(M,2)^2;
    disp('Peak Power:');
    disp(ppeak);
    % Plot Unfiltered Data
    
    % subplot(212);
    plot(faxis,mag,'LineWidth',1.5); hold on;
    title('FFT','FontSize' ,10);
    xlabel('Frequency (RPM)','FontSize' ,10);
    ylabel('Magnitude','FontSize' ,10);
    
    

    
    
end
legend('User 5','User 4','User 3','User 2','User 1');
hold off;
% Time length of signal



% Filtered Data
cutfreq = [.25 15];
[b,a] = butter(4,cutfreq./(Fs./2));
lpf = filter(b,a,data)  ;
bdata = fft(lpf,fftlength) / L;
[~,idx] = max(abs(bdata(1:ctr)));
% bdata(idx) = 0;
fftcadence = faxis(idx);
powah = bandpower(lpf,Fs,[fftcadence/60-.001 fftcadence/60+.001]);
totpowah = bandpower(lpf);
% 
% plot(faxis,abs(bdata(1:ctr)),'linewidth',1);
% legend('unfilt','butter LPF'); hold off;


% figure;
% subplot 221;
    % windlen = floor(length(lpf)/10);
%     windt = 4;
%     windlen = floor(windt*Fs);
%     nlap= [];
%     nfft=2^nextpow2(windlen);
%     wind = hamming(windlen);
%     spectrogram(lpf,wind,nlap,nfft,Fs,'yaxis');hold on;
%     title('Spectrogram (4s)');
%     % colorbar;
%     %plot(t,f(I),q,'r','linewidth',2); 
%     hold off;
%     
% 
% 
% subplot 222;
%     [s,f,t,pxx]=spectrogram(lpf,wind,nlap,nfft,Fs,'yaxis');
% %     disp(['Spectrogram time: ', num2str(t(length(t))),' seconds.']);
%     [~,I] = max(10*log10(pxx)); % largest PSD in each column (STFT).
%     cadot = 60*f(I); %cadence over time
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
%     [~,f,t,pxx]=spectrogram(lpf,wind,nlap,nfft,Fs,'yaxis');
%     [M,I] = max(10*log10(pxx)); % largest PSD in each column (STFT).
%     M
%     cadot = 60*f(I);
%     cadlen = length(cadot);
%     x = linspace(0,t(length(t)),cadlen);
%     stairs(x,cadot,'linewidth',2);
%     grid on;
%     title('Discrete Cadence Over Time');
%     xlabel('Time (mins)');
%     ylabel('Cadence (RPM)');









% avgcadence = mean(cadot);

% save output over time to .csv file
% fname = strsplit(filename,'.csv');
% newfilename = sprintf('%speaks.csv',fname{1});
% q = [x',cadot];
% disp(['Window Length: ',num2str(windlen)]);
% disp(['FFT Peak: ',num2str(fftcadence)]);
% disp(['STFT Avg: ', num2str(avgcadence)]);
% csvwrite(newfilename,q);


end
