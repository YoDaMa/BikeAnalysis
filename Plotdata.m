function Plotdata(filename)

close all;

Fs = 100;

%% Importing and Formatting the Data
SENS_DATA = importdata(filename);
parsedata = SENS_DATA.data;
dlen = length(parsedata(:,1));
fdel = 20;
edel = 20;
SENS_DATA = parsedata(fdel*Fs:dlen-edel*Fs,:);
% SENS_DATA = parsedata(1:timedel*Fs,:);
timestamp = SENS_DATA(:,1);
ATT_R = SENS_DATA(:,2);
ATT_P = SENS_DATA(:,3);
ATT_Y = SENS_DATA(:,4);
ROTR_X = SENS_DATA(:,5);
ROTR_Y = SENS_DATA(:,6);
ROTR_Z = SENS_DATA(:,7);
GR_X = SENS_DATA(:,8);
GR_Y = SENS_DATA(:,9);
GR_Z = SENS_DATA(:,10);
UA_X = SENS_DATA(:,11);
UA_Y = SENS_DATA(:,12);
UA_Z = SENS_DATA(:,13);
MAG_X = SENS_DATA(:,12);
MAG_Y = SENS_DATA(:,13);
MAG_Z = SENS_DATA(:,14);


keep_rows = find(~isnan(UA_X)); % find the 1s produced by isnan.

% Timestamp
ts = timestamp(keep_rows);
ts = ts - ts(1);
[~,sorted] = sort(ts); % get indexes of sorted times

% Attitude 
ATT_R = ATT_R(keep_rows);
ATT_P = ATT_P(keep_rows);
ATT_Y = ATT_Y(keep_rows);
ATT_R = ATT_R(sorted);
ATT_P = ATT_P(sorted);
ATT_Y = ATT_Y(sorted);
attitude = [ATT_R,ATT_P,ATT_Y];

% Rotation Rate
ROTR_X = ROTR_X(keep_rows);
ROTR_Y = ROTR_Y(keep_rows);
ROTR_Z = ROTR_Z(keep_rows);
ROTR_X = ROTR_X(sorted);
ROTR_Y = ROTR_Y(sorted);
ROTR_Z = ROTR_Z(sorted);
rotation = [ROTR_X,ROTR_Y,ROTR_Z];

% Gravitational Acceleration
GR_X = GR_X(keep_rows);
GR_Y = GR_Y(keep_rows);
GR_Z = GR_Z(keep_rows);
GR_X = GR_X(sorted);
GR_Y = GR_Y(sorted);
GR_Z = GR_Z(sorted);
gravity = [GR_X,GR_Y,GR_Z];


% User acceleration
Accel_X = UA_X(keep_rows);
Accel_Y = UA_Y(keep_rows);
Accel_Z = UA_Z(keep_rows);
Accel_X = Accel_X(sorted);
Accel_Y = Accel_Y(sorted);
Accel_Z = Accel_Z(sorted);
accel = [Accel_X Accel_Y Accel_Z];

% Magnietic Field 
MAG_X = MAG_X(keep_rows);
MAG_Y = MAG_Y(keep_rows);
MAG_Z = MAG_Z(keep_rows);
MAG_X = MAG_X(sorted);
MAG_Y = MAG_Y(sorted);
MAG_Z = MAG_Z(sorted);
magnetic = [MAG_X,MAG_Y,MAG_Z];


%% Plotting Signals
ns = length(Accel_X);
t = linspace(0,ns/Fs,ns);

% Attitude
figure;
attitude = plot(t,ATT_R,t,ATT_P,t,ATT_Y);
xlabel('Time (s)');
ylabel('N/A');
title('Attitude');

% Rate of Rotation
figure;
rotation = plot(t,ROTR_X,t,ROTR_Y,t,ROTR_Z);
xlabel('Time (s)');
ylabel('Rate (rad/s)'); % I am not entirely sure on these units
title('Rate of Rotation');

% Gravitational Acceleration
figure;
gravity = plot(t,GR_X,t,GR_Y,t,GR_Z);
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
title('Gravitational Acceleration');

% User Acceleration
figure;
acceleration = plot(t,Accel_X,t,Accel_Y,t,Accel_Z);
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
title('User Acceleration');

% Magnetism
figure('Name','Magnetism','NumberTitle','off');
plot(t,MAG_X); hold on;
plot(t,MAG_Y);
plot(t,MAG_Z);
xlabel('Time (s)');
ylabel('Magnetic Flux Density (uT)');
title('Magnetism');
hold off;








%% ETC
% Time length of signal
sigtime = ns/Fs/60.0;
writing = sprintf('\nThe signal length is %d minutes.\n',sigtime);
disp(writing);


powX = bandpower(Accel_X); % Find the power contained between 30 and 40s of Acceleration data after removing 20s noise.
powY = bandpower(Accel_Y);
powZ = bandpower(Accel_Z);
powers = [powX powY powZ];
ptot = sum(powers);
[~,powidx] = max(powers);

npowX = bandpower(Accel_X(1:11*Fs));
npowY = bandpower(Accel_Y(1:11*Fs));
npowZ = bandpower(Accel_Z(1:11*Fs));


end