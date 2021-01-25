%% Project
%
% Step 1:
% Import Data and clean/filter
%%
clear all; close all;
%% ECG .dat file reader (from mathworks.com/matlabcentral/fileexchange/49822-open_ecg-ecg-dat-file-reader)
%path_file = fullfile(pwd, 'group project\data\mit-bih-arrhythmia-database-1.0.0' , '100.dat');

%%% Choose file %%%
%[filename, pathname] = uigetfile('*.dat', 'Open file .dat');
%if isequal(filename, 0) || isequal(pathname, 0)
%        disp('file input canceled.');
%        ECG_Date = [];
%else
%    fid = fopen(filename, 'r')
%end;

fid = fopen(fullfile('data', 'mit-bih-arrhythmia-database-1.0.0', '101.dat'), 'r');
time = 15;
f=fread(fid, 2*360*time, 'ubit12');
Orig_Sig = f(1:2:length(f));
figure;
plot(Orig_Sig, 'ro');
hold on;
plot(Orig_Sig, 'b');
hold off;

%% Filtering: Transform to fourier domain
% ECG ~ 0.15 to 150HZ (American Heart Association) *from lecture slides
% band-pass filter the signal between 0.15-150Hz
% sample with at least 300Hz

ftECG = fft(Orig_Sig);

%% Plot
% Notice that the data has a spike on at the very start and one in the
% middle.
plot(abs(ftECG))

%% Plot shifted data
plot(abs(fftshift(ftECG)))

%% Window Size
nz = 20;
lenEcg = size(Orig_Sig, 1);

%% Create filter vector and applies it to ft of ECG
midOnes = ones(1,(lenEcg-(2*nz)));

lenEcg - size(midOnes,2)

flt = [zeros(1,nz), midOnes, zeros(1,nz)];

size(flt)
size(ftECG)

ftecg_hp = ftECG' .* flt;

figure; semilogy(abs(ftECG), 'b', 'linew', 3); 
hold on;
semilogy(abs(ftecg_hp),'r')
legend('original', 'after high pass');

%%

% Plot using fftshift...
figure; plot(abs(fftshift(ftecg_hp))); xlabel('frequency'); ylabel('magnitude of fourier components');
hold on; plot(10^4*fftshift(flt),'r','linew',2)
legend('Filtered fourier','high pass filter')

%% Inverse FFT

ecg_hp = real( ifft( ftecg_hp));

figure;
% Plot the original ecg signal.
%plot(Orig_Sig,'r');
hold on;
% Plot the ecg signal in the time domain after we removed some frequencies.
plot(ecg_hp,'b');
xlabel('time'); ylabel('electrical activity'); grid
legend('original ecg','ecg with low frequencies removed');
