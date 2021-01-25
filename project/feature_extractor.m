%% Project
% Step 1: 
% Data acquisition 
%
% Step 2:
% Feature extraction
%
clear all; close all;
%% ECG .dat file reader (from mathworks.com/matlabcentral/fileexchange/49822-open_ecg-ecg-dat-file-reader)
% Extract data from the .dat files from the dataset.

fid = [];
fileNum = [100:109, 111:119, 121:124, 200:203, 205, 207:210, 212:215, 217, 219:223, 228, 230:234];
fileLoc = fullfile('data', 'mit-bih-arrhythmia-database-1.0.0/');
numOfTest = length(fileNum);
for i = 1:numOfTest
    fid(end+1) = fopen(strcat(fileLoc,num2str(fileNum(i)),'.dat'),'r');
end

time = 15;
numTests = 100;
ecg = [];

for i = 1:numOfTest
    
    f=fread(fid(i), numTests * 2*360*time, 'ubit12');
    for k = 1:numTests
        
        range = 2*360*time * (k - 1) + 1 : 2 : 2*360*time * k;
        Orig_Sig = f(range);
        %plot(Orig_Sig);

        ftECG = fft(Orig_Sig);

        nz = 20;
        lenEcg = size(Orig_Sig, 1);

        midOnes = ones(1,(lenEcg-(2*nz)));
        endOnes = ones(1, ((lenEcg-nz)/2));
        flt1 = [zeros(1,nz), midOnes, zeros(1,nz)];
        flt2 = [endOnes, zeros(1,nz), endOnes];
        flt = flt1 .* flt2;

        ftecg_hp = ftECG' .* flt;
        ecg(((i - 1) * numTests) + k, :) = real( ifft( ftecg_hp));
    %     figure;
    %     plot(ecg_hp,'b');
    %     xlabel('time'); ylabel('electrical activity'); grid
    %     legend('ecg with low frequencies removed');
    end
end 

%% Feature Extraction - Frist-Order
% Extract some frist order features from the data.
% Use "format shorte" to see non-zeros for small values.

all_features = [];
for i = 1:numOfTest
    
    for k = 1:numTests
        all_features(((i - 1) * numTests) + k,1) = mean(ecg(((i - 1) * numTests) + k,:));

        % Variance
        all_features(((i - 1) * numTests) + k,2) = var(ecg(((i - 1) * numTests) + k,:));

        % Standard Deviation
        all_features(((i - 1) * numTests) + k,3) = std(ecg(((i - 1) * numTests) + k,:));

        % Mean Deviation
        all_features(((i - 1) * numTests) + k,4) = 1/length(ecg(((i - 1) * numTests) + k,:)) * (sum(ecg(((i - 1) * numTests) + k,:) - mean(ecg(((i - 1) * numTests) + k,:))));

        % Skewness
        all_features(((i - 1) * numTests) + k,5) = skewness(ecg(((i - 1) * numTests) + k,:));

        % Kurtosis
        all_features(((i - 1) * numTests) + k,6) = kurtosis(ecg(((i - 1) * numTests) + k,:));
        
        % Mean QR-interval and mean QS-interval
        plotGraph = false;
        [mean_QR_interval, mean_QS_interval] = find_QRS_features(ecg(((i - 1) * numTests) + k,:), plotGraph);
        all_features(((i - 1) * numTests) + k,7) = mean_QR_interval;
        all_features(((i - 1) * numTests) + k,8) = mean_QS_interval;
        
        % Label
        if i <= 23
            all_features(((i - 1) * numTests) + k,9) = 0;
        else
            all_features(((i - 1) * numTests) + k,9) = 1;
        end
        
    end

end

%% Export
% export the features into a csv file.
writematrix(all_features, 'norm_feature.csv');

%% Plot the first ECG with its QRS points labelled
% Shows that the finding QRS points function is working

find_QRS_features(ecg(1,:), true);

%% Function to find the QRS features

function [mean_QR_interval, mean_QS_interval] = find_QRS_features(ecg_data, plotGraph)
% find_QRS_features finds the mean QR- and QS-intervals for the ECG.
% plotGraph: logical value, representing if we will plot the ECG with the
% QRS points.

    %% Local max/mins
    %f=fread(fid(1), 2*360*time, 'ubit12');
    %ecg_data = f(1:2:length(f));

    x = [1:1:length(ecg_data)];

    [maxVal, maxIndex] = findpeaks(ecg_data,x);
    localMin = islocalmin(ecg_data);

    %% Plot all local mins/maxes
    if plotGraph
        figure;
        plot(x, ecg_data);
        hold on;
        plot(maxIndex, maxVal, "ro");
        plot(x(localMin), ecg_data(localMin), 'go');
        hold off;
    end

    %% get QRS

    % get R
    max_threshold  = max(ecg_data) - (max(ecg_data)-min(ecg_data))/2;
    R = zeros(size(ecg_data));
    for i = 1:length(maxVal)
        if maxVal(i) > max_threshold
            R(maxIndex(i)) = 1;
        end 
    end
    R = logical(R);

    % get Q/S using R
    minIndex = [];
    r_index = [];
    for i = 1:length(localMin)
        if localMin(i) == 1
            minIndex(end+1) = i;
        end
        if R(i) == 1
            r_index(end+1) = i;
        end
    end

    prevMinIndex = minIndex(1);
    Q = zeros(size(ecg_data));
    S = zeros(size(ecg_data));
    rPeakIndex = 1;
    for i = 2:length(minIndex)
        if r_index(rPeakIndex) < minIndex(i)
            Q(minIndex(i-1)) = 1;
            S(minIndex(i)) = 1;
            rPeakIndex = rPeakIndex + 1;
            if rPeakIndex > length(r_index)
                break;
            end
        end
    end
    Q = logical(Q);
    S = logical(S);

    q_index = [];
    r_index;
    s_index = [];

    for i = 1:length(Q)
        if Q(i) == 1
            q_index(end+1) = i;
        end
        if S(i) == 1
            s_index(end+1) = i;
        end
    end

    %% Plot to show QRS on ECG
    if plotGraph
        plot(x, ecg_data);
        hold on;
        plot(x(R), ecg_data(R), 'ro');
        plot(x(Q), ecg_data(Q), 'bo');
        plot(x(S), ecg_data(S), 'go');
        legend('ECG', 'R', 'Q', 'S');
        hold off;
    end

    %% Features from QRS
    qVal = ecg_data(Q);
    rVal = ecg_data(R);
    sVal = ecg_data(S);

    %QRS_features = []
    % Mean of Q, R, S
    %qMean = mean(qVal);
    %rMean = mean(rVal);
    %sMean = mean(sVal);

    % Variance of Q, R, S
    %qVar = var(qVal);
    %rVar = var(rVal);
    %sVar = var(sVal);

    % Standard Deviation Q, R, S
    %qStd = sqrt(qVar);
    %rStd = sqrt(rVar);
    %sStd = sqrt(sVar);

    % Difference from R
    rqDiff = zeros(size(q_index));

    % Difference from Q to S (length/duration of peak)
    sqDiff = zeros(size(q_index));

    for i = 1:length(q_index)
        rqDiff(i) = r_index(i) - q_index(i);
        sqDiff(i) = s_index(i) - q_index(i);
    end

    mean_QR_interval = mean(rqDiff);
    mean_QS_interval = mean(sqDiff);
end