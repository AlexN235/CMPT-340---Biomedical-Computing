


%% Get the Data
T = readtable('norm_feature.csv');
T = T{:,:};

T2 = T(randperm(size(T, 1)), :);

%% Get the training and testing set
training = T2(1:size(T2, 1) * 0.8, :);
testing = T2(size(T2, 1) * 0.8 + 1:end, :);

%% Create the models and predictions
num_neighbors = 7;
num_trees = 200;
features = 1:8;
labelCol = 9;

knnMdl = fitcknn(training(:, features), training(:, labelCol), 'NumNeighbors', num_neighbors);           % k nearest neighbors
nbMdl = fitcnb(training(:, features), training(:, labelCol));                                            % Naive Bayes
dtMdl = fitctree(training(:, features), training(:, labelCol));                                          % Decision Tree
rfMdl = TreeBagger(num_trees , training(:, features),training(:, labelCol), 'Method', 'classification'); % Random Forest                          

knnP = predict(knnMdl, testing(:, features));
nbP = predict(nbMdl, testing(:, features));
dtP = predict(dtMdl, testing(:, features));
rfP = str2double(predict(rfMdl, testing(:, features)));

%% Check accuracy
correctKnn = 0;
correctNb = 0;
correctDt = 0;
correctRf = 0;
for i = 1:size(testing, 1)
    
    if knnP(i) == testing(i, labelCol)
        correctKnn = correctKnn + 1;
    end
    
    if nbP(i) == testing(i, labelCol)
        correctNb = correctNb + 1;
    end
    
    if dtP(i) == testing(i, labelCol)
        correctDt = correctDt + 1;
    end
    
    if rfP(i) == testing(i, labelCol)
        correctRf = correctRf + 1;
    end
    
end

fprintf('Naive Bayes Accuracy: %f \n', correctNb / size(testing, 1));
fprintf('K neighbors Accuracy: %f \n', correctKnn / size(testing, 1));
fprintf('Decision Tree Accuracy: %f \n', correctDt / size(testing, 1));
fprintf('Random Forest Accuracy: %f \n', correctRf / size(testing, 1));


%% Display Confusion Matrix
figure (1);
confusionchart(testing(:, labelCol), knnP);
title("K Nearest Neighbor Classifier")

figure (2);
confusionchart(testing(:, labelCol), nbP);
title("Naive Bayes Classifier")

figure (3);
confusionchart(testing(:, labelCol), dtP);
title("Decision Tree Classifier")

figure (4);
confusionchart(testing(:, labelCol), rfP);
title("Random Forest Classifier")

%% Calculate False postives and False negatives, True postives, True negatives

% Knn
FNKnn = numel(knnP(testing(:, 9) == 1 & knnP == 0));
FPKnn = numel(knnP(testing(:, 9) == 0 & knnP == 1));
TPKnn = numel(knnP(testing(:, 9) == 1 & knnP == 1));
TNKnn = numel(knnP(testing(:, 9) == 0 & knnP == 0));

% Naive Bayes
FNNb = numel(nbP(testing(:, 9) == 1 & nbP == 0));
FPNb = numel(nbP(testing(:, 9) == 0 & nbP == 1));
TPNb = numel(nbP(testing(:, 9) == 1 & nbP == 1));
TNNb = numel(nbP(testing(:, 9) == 0 & nbP == 0));

% Decision tree
FNDt = numel(dtP(testing(:, 9) == 1 & dtP == 0));
FPDt = numel(dtP(testing(:, 9) == 0 & dtP == 1));
TPDt = numel(dtP(testing(:, 9) == 1 & dtP == 1));
TNDt = numel(dtP(testing(:, 9) == 0 & dtP == 0));

% Random forest
FNRf = numel(rfP(testing(:, 9) == 1 & rfP == 0));
FPRf = numel(rfP(testing(:, 9) == 0 & rfP == 1));
TPRf = numel(rfP(testing(:, 9) == 1 & rfP == 1));
TNRf = numel(rfP(testing(:, 9) == 0 & rfP == 0));

%% Calculate False positve and False negative rates

FNRKnn = FNKnn / (TNKnn + FPKnn);
TNRKnn = 1 - FNRKnn;
FPRKnn = FPKnn / (TPKnn + FNKnn);
TPRKnn = 1 - FPRKnn;

FNRNb = FNNb / (TNNb + FPNb);
TNRNb = 1 - FNRNb;
FPRNb = FPNb / (TPNb + FNNb);
TPRNb = 1 - FPRNb;

FNRDt = FNDt / (TNDt + FPDt);
TNRDt = 1 - FNRDt;
FPRDt = FPDt / (TPDt + FNDt);
TPRDt = 1 - FPRDt;

FNRRf = FNRf / (TNRf + FPRf);
TNRRf = 1 - FNRRf;
FPRRf = FPRf / (TPRf + FNRf);
TPRRf = 1 - FPRRf;



fprintf('\nNaive Bayes False Negative Rate: %f \n', FNRNb);
fprintf('Naive Bayes False Positive Rate: %f \n', FPRNb);
fprintf('K neighbors False Negative Rate: %f \n', FNRKnn);
fprintf('K neighbors False Positive Rate: %f \n', FPRKnn);
fprintf('Decision Tree False Negative Rate: %f \n', FNRDt);
fprintf('Decision Tree False Positive Rate: %f \n', FPRDt);
fprintf('Random Forest False Negative Rate: %f \n', FNRRf);
fprintf('Random Forest False Positive Rate: %f \n', FPRRf);





