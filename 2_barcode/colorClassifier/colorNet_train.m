clear

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

temp_layer = fullyConnectedLayer(3, 'Name', 'fc3');
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
%%
ds = imageDatastore('./database/R');
reds = zeros(224, 224, 3, 135*8, "uint8");
for i = 1 : size(ds.Files, 1)
    img = ds.read();
    reds(:, :, :, (i-1)*8 + 1) = img;
    reds(:, :, :, (i-1)*8 + 2) = imrotate(img, 90);
    reds(:, :, :, (i-1)*8 + 3) = imrotate(img, 180);
    reds(:, :, :, (i-1)*8 + 4) = imrotate(img, 270);
    reds(:, :, :, (i-1)*8 + 5) = flip(img, 1);
    reds(:, :, :, (i-1)*8 + 6) = imrotate(flip(img, 1), 90);
    reds(:, :, :, (i-1)*8 + 7) = imrotate(flip(img, 1), 180);
    reds(:, :, :, (i-1)*8 + 8) = imrotate(flip(img, 1), 270);
end

ds = imageDatastore('./database/Y');
yellows = zeros(224, 224, 3, 133*8, "uint8");
for i = 1 : size(ds.Files, 1)
    img = ds.read();
    yellows(:, :, :, (i-1)*8 + 1) = img;
    yellows(:, :, :, (i-1)*8 + 2) = imrotate(img, 90);
    yellows(:, :, :, (i-1)*8 + 3) = imrotate(img, 180);
    yellows(:, :, :, (i-1)*8 + 4) = imrotate(img, 270);
    yellows(:, :, :, (i-1)*8 + 5) = flip(img, 1);
    yellows(:, :, :, (i-1)*8 + 6) = imrotate(flip(img, 1), 90);
    yellows(:, :, :, (i-1)*8 + 7) = imrotate(flip(img, 1), 180);
    yellows(:, :, :, (i-1)*8 + 8) = imrotate(flip(img, 1), 270);
end


ds = imageDatastore('./database/B');
blues = zeros(224, 224, 3, 135*8, "uint8");
for i = 1 : size(ds.Files, 1)
    img = ds.read();
    blues(:, :, :, (i-1)*8 + 1) = img;
    blues(:, :, :, (i-1)*8 + 2) = imrotate(img, 90);
    blues(:, :, :, (i-1)*8 + 3) = imrotate(img, 180);
    blues(:, :, :, (i-1)*8 + 4) = imrotate(img, 270);
    blues(:, :, :, (i-1)*8 + 5) = flip(img, 1);
    blues(:, :, :, (i-1)*8 + 6) = imrotate(flip(img, 1), 90);
    blues(:, :, :, (i-1)*8 + 7) = imrotate(flip(img, 1), 180);
    blues(:, :, :, (i-1)*8 + 8) = imrotate(flip(img, 1), 270);
end
%%
trainImages = zeros(224, 224, 3, 3224, "uint8");
trainImages(:, :, :, 1:135*8) = reds;
trainImages(:, :, :, 135*8+1:268*8) = yellows;
trainImages(:, :, :, 268*8+1:3224) = blues;
trainGt = categorical([repmat("red", 135*8, 1); repmat("yellow", 133*8, 1); repmat("blue", 135*8, 1);]); 
%%
len = size(trainImages, 4);
idx = randperm(len, floor(len * 0.15));
validImages = trainImages(:, :, :, idx);
validGt = trainGt(idx);
trainImages(:, :, :, idx) = [];
trainGt(idx) = [];
%%
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 5, ...
    'LearnRateDropFactor', 0.7, ...
    'SquaredGradientDecayFactor', 0.99, ...
    'L2Regularization', 0.0001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ValidationData', {validImages, validGt}, ...
    'ValidationFrequency', 30, ...
    'ValidationPatience', 5, ...
    'ExecutionEnvironment', 'gpu' ...
    );

colorNet = trainNetwork(trainImages, trainGt, lgraph, options);