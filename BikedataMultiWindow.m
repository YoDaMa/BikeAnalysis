function BikedataMultiFile(filename) 

close all;

Fs = 100; % Visually inspecting the oscilloscope
%filename = 'Bike_1.csv'
SENS_DATA = importdata(filename);
parsedata = SENS_DATA.data;
dlen = length(parsedata(:,1));
fdel = 10;
edel = 10;
SENS_DATA = parsedata(fdel*Fs:dlen-edel*Fs,:);
% SENS_DATA = parsedata(1:timedel*Fs,:);
timestamp = SENS_DATA(:,1);
UA_X = SENS_DATA(:,11);
UA_Y = SENS_DATA(:,12);
UA_Z = SENS_DATA(:,13);


keep_rows = find(~isnan(UA_X)); % find the 1s produced by isnan.

% User acceleration
Accel_X = UA_X(keep_rows);
Accel_Y = UA_Y(keep_rows);
Accel_Z = UA_Z(keep_rows);
ts = timestamp(keep_rows);
ts = ts - ts(1);
[~,sorted] = sort(ts);
Accel_X1 = Accel_X(sorted);
rateplot = [];
cadot = [];
powpow = [];

enditr =floor(length(Accel_X1)/10)-20;
disp(enditr);
figure;
for i = linspace(1,301,31) 
    disp('Start Time:');
    ws = i
    we = i+20;
    Accel_X = Accel_X1(ws*Fs:we*Fs);
    sigtime = length(Accel_X)/Fs/60.0;
%     writing = sprintf('\nThe signal length is %d minutes.\n',sigtime); 
%     disp(writing);


    t = linspace(0,length(Accel_X)/Fs,length(Accel_X));
    hold on;
    xlabel('Time (sec)', 'FontSize' ,10);
    ylabel('User Acceleration (m/s^2)', 'FontSize', 10);
    title('Cadence of Bike User','FontSize',10);
    data = Accel_X;
    data = data-mean(data);% subtract DC value
    plot(t,data,'linewidth',2); 
    % findpeaks(data, Fs,'MinPeakDistance',1/100,'MinPeakHeight',0);

    % legend('X','Y','Z','Selected');
    
    fftlength = 2^(nextpow2(length(data))+3);
    L = length(data);
    ctr = (fftlength / 2) + 1; 
    faxis = 60*(Fs / 2) .* linspace(0,1, ctr);    
    cutfreq = [.25 15];
    [b,a] = butter(4,cutfreq./(Fs./2));
    lpf = filter(b,a,data);
    bdata = fft(lpf,fftlength) / L;
    [M,idx] = max(abs(bdata(1:ctr)));

    % bdata(idx) = 0;
    disp('Cadence:');
    fftcadence = faxis(idx)
    cadot = [cadot,fftcadence];
    disp('Powers:');
    powah = bandpower(lpf,Fs,[fftcadence/60-.001 fftcadence/60+.001])
    powah = norm(M,2)^2
    totpowah = bandpower(lpf)
    rate = powah/totpowah
    powpow = [powpow,totpowah];
    rateplot = [rateplot,rate];

end
hold off;
disp('Finally...')
disp(faxis(idx))
figure;
T = 1/Fs;
L = 1000;
t = (0:T-1)*T;
length(t)
y = sin(2*pi*faxis(idx).*t) ;
plot(t,y);
fftlength = 2^10;
f = fft(y,fftlength)/length(t);
figure;
plot(abs(f));


donedata = [cadot;rateplot;powpow];
cadot
powpow
rateplot
csvwrite('Multiwindow.csv',donedata);


end
