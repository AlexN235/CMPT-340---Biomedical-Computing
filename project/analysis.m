%% Project
%
% Step 4: Analysis using classifiers (ROC, SVM, CNN, ANN)
% Step 4.1: With Linear Discriminant Analysis
% Step 4.2: With SVM
clear all; close all;
%% Get the Data
features = 1:8;
T = readtable('norm_feature.csv');
T = T{:,features};

%% Split data into positives and negatives
numTests = 100;
negatives = 23;
rows = numTests * negatives;
data_pos = T(rows + 1:2 * rows, :)';
data_neg = T(1:rows, :)';

%% Compute Mean and Covariance of each feature for each classification
mean_pos = mean(data_pos);
mean_neg = mean(data_neg);

cov_pos = cov(data_pos);
cov_neg = cov(data_neg);

%% Compute v from the mean and covariance
v = ((cov_pos+cov_neg) \ (mean_pos - mean_neg)');
%% Project into 1D vectors
data_pos_1D = data_pos*v;
data_neg_1D = data_neg*v;
%% Fit into a guassian distribution
mean_pos_1D = mean(data_pos_1D);
std_pos_1D = std(data_pos_1D);

mean_neg_1D = mean(data_neg_1D);
std_neg_1D = std(data_neg_1D);

std_devs = 4;
x = mean_neg_1D - std_devs * std_neg_1D:0.001:mean_pos_1D + std_devs * std_pos_1D;

figure(1)
% We use the normal distribution.
negY = 1/(std_neg_1D* sqrt(2*pi)) * exp(- (((x-mean_neg_1D).^2)./(2*std_neg_1D^2)));
posY = 1/(std_pos_1D* sqrt(2*pi)) * exp(- (((x-mean_pos_1D).^2)./(2*std_pos_1D^2)));

h1 = plot(x, negY,'.b');
hold on;
h2 = plot(x, posY,'.r');
hold off;
xlabel('data projected onto v')
ylabel('probability')
title('Gaussian distributions')

%% Threshold for ROC curve and ROC curve
if mean_pos_1D <= mean_neg_1D
    T= linspace(mean_neg_1D-5,mean_pos_1D+5,1000);
else
    T= linspace(mean_pos_1D-5,mean_neg_1D+5,1000);
end
%% Choose best threshold
TN = normcdf(T, mean_neg_1D, std_neg_1D);
FN = normcdf(T, mean_pos_1D, std_pos_1D);

figure(2);
plot(FN, TN);
xlabel('FN')
ylabel('TN')
title("ROC for FN vs TN")

TP = 1-FN;
both = TN + TP;
max_val = max(both);
threshIdx = find(both == max_val);
%% Classify using threashold
threshold = T(threshIdx);
correctEarly = 0;
correctAdvanced = 0;
for i = 1:length(data_pos_1D)
    if data_pos_1D(i) > threshold
        correctAdvanced = correctAdvanced + 1;
    end
end
for i = 1:length(data_pos_1D)
    if data_neg_1D(i) < threshold
        correctEarly = correctEarly + 1;
    end
end

correctAdvanced
correctEarly

%% Plot the threshold on the figures above
figure(2);
hold on;
plot(FN(threshIdx), TN(threshIdx), '*r', 'MarkerSize', 20);

figure(1);
hold on;
h3 = plot(threshold, (0:0.001:0.25), '.k');
legend([h1, h2, h3(1)], 'negative', 'positive', 'chosen threshold')