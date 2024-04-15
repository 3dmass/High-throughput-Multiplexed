clear all
%%

res = 224;
level = 50;

if level == 18
    net = resnet18('Weights', 'imagenet');
elseif level == 50
    net = resnet50('Weights', 'imagenet');
elseif level == 101
    net = resnet101('Weights','imagenet');
end


lgraph = layerGraph(net);

temp_layer = fullyConnectedLayer(30, 'Name', 'fc3');
lgraph = replaceLayer(lgraph, 'fc1000', temp_layer);

if level == 18
    temp_layer = classificationLayer('Name', 'output');
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', temp_layer);
elseif level == 50
    temp_layer = classificationLayer('Name', 'output');
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', temp_layer);
elseif level == 101
    temp_layer = classificationLayer('Name', 'output');
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', temp_layer);
end

temp_layer = imageInputLayer([56 56 3], 'Name', 'input');
lgraph = replaceLayer(lgraph, 'input_1', temp_layer);
%%
trainset = csvread("dataset/train.csv");
testset = csvread("dataset/test.csv");


X_train_temp = trainset(:, 1:3);
X_train = ones(56, 56, 3, size(trainset, 1), 'double');
for i = 1 : size(trainset, 1)
    X_train(:, :, 1, i) = X_train(:, :, 1, i) * X_train_temp(i, 1) / 255;
    X_train(:, :, 2, i) = X_train(:, :, 2, i) * X_train_temp(i, 2) / 255;
    X_train(:, :, 3, i) = X_train(:, :, 3, i) * X_train_temp(i, 3) / 255;
end
Y_train = string(int32(trainset(:, 4:5)));
mask = Y_train(:, 1) == "0"; Y_train(mask) = "R";
mask = Y_train(:, 1) == "1"; Y_train(mask) = "Y";
mask = Y_train(:, 1) == "2"; Y_train(mask) = "B";
Y_train = Y_train(:, 1) + Y_train(:, 2);
Y_train = categorical(Y_train);

X_test_temp = testset(:, 1:3);
X_test = ones(56, 56, 3, size(testset, 1), 'double');
for i = 1 : size(testset, 1)
    X_test(:, :, 1, i) = X_test(:, :, 1, i) * X_test_temp(i, 1) / 255;
    X_test(:, :, 2, i) = X_test(:, :, 2, i) * X_test_temp(i, 2) / 255;
    X_test(:, :, 3, i) = X_test(:, :, 3, i) * X_test_temp(i, 3) / 255;
end
Y_test = string(int32(testset(:, 4:5)));
mask = Y_test(:, 1) == "0"; Y_test(mask) = "R";
mask = Y_test(:, 1) == "1"; Y_test(mask) = "Y";
mask = Y_test(:, 1) == "2"; Y_test(mask) = "B";
Y_test = Y_test(:, 1) + Y_test(:, 2);
Y_test = categorical(Y_test);
%%
X = cat(4, X_train, X_test);
Y = [Y_train; Y_test];
pm = randperm(215802);
percentage = floor(215802 * 0.3);
X_train = X;
X_test = X(:, :, :, pm(1:percentage));
X_train(:, :, :, pm(1:percentage)) = [];

Y_train = Y;
Y_test = Y(pm(1:percentage));
Y_train(pm(1:percentage)) = [];
%%
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.00001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 5, ...
    'LearnRateDropFactor', 0.7, ...
    'SquaredGradientDecayFactor', 0.99, ...
    'L2Regularization', 0.0001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 1024, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ValidationData', {X_test, Y_test}, ...
    'ValidationFrequency', 100, ...
    'ValidationPatience', 5, ...
    'ExecutionEnvironment', 'gpu' ...
    );

colorNet = trainNetwork(X_train, Y_train, lgraph, options);
%%
pred = classify(colorNet, X_test);
%%
cm = confusionchart(pred, Y_test);
cm.Title = "Confusion Matrix";
cm.XLabel = "Predicted Class";
cm.YLabel = "True Class";
cm.RowSummary = "row-normalized";
cm.ColumnSummary = "column-normalized";
cm.FontSize = 17;

% plotconfusion(pred, Y_test);
%%
save("network_1.mat", "colorNet", "pred");
%%
save("network_2.mat", "X_test", "X_train", "Y_train", "Y_test", "-v7.3");
