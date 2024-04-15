clear

res = 224;

net = resnet50('Weights', 'imagenet');

lgraph = layerGraph(net);

temp_layer = fullyConnectedLayer(5, 'Name', 'fc5');
lgraph = replaceLayer(lgraph, 'fc1000', temp_layer);

temp_layer = classificationLayer('Name', 'output');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', temp_layer);

%%
ds = imageDatastore('circle');
circles = zeros(224, 224, 3, 97*8, "uint8");
for i = 1 : size(ds.Files, 1)
    img = imresize(imadjust(ds.read()), [224 224]);
    img = cat(3, img, img,img);
    circles(:, :, :, (i-1)*8 + 1) = img;
    circles(:, :, :, (i-1)*8 + 2) = imrotate(img, 90);
    circles(:, :, :, (i-1)*8 + 3) = imrotate(img, 180);
    circles(:, :, :, (i-1)*8 + 4) = imrotate(img, 270);
    circles(:, :, :, (i-1)*8 + 5) = flip(img, 1);
    circles(:, :, :, (i-1)*8 + 6) = imrotate(flip(img, 1), 90);
    circles(:, :, :, (i-1)*8 + 7) = imrotate(flip(img, 1), 180);
    circles(:, :, :, (i-1)*8 + 8) = imrotate(flip(img, 1), 270);
end

ds = imageDatastore('cross');
crosses = zeros(224, 224, 3, 100*8, "uint8");
for i = 1 : size(ds.Files, 1)
    img = imresize(imadjust(ds.read()), [224 224]);
    img = cat(3, img, img,img);
    crosses(:, :, :, (i-1)*8 + 1) = img;
    crosses(:, :, :, (i-1)*8 + 2) = imrotate(img, 90);
    crosses(:, :, :, (i-1)*8 + 3) = imrotate(img, 180);
    crosses(:, :, :, (i-1)*8 + 4) = imrotate(img, 270);
    crosses(:, :, :, (i-1)*8 + 5) = flip(img, 1);
    crosses(:, :, :, (i-1)*8 + 6) = imrotate(flip(img, 1), 90);
    crosses(:, :, :, (i-1)*8 + 7) = imrotate(flip(img, 1), 180);
    crosses(:, :, :, (i-1)*8 + 8) = imrotate(flip(img, 1), 270);
end

ds = imageDatastore('hexagonal');
hexagonals = zeros(224, 224, 3, 108*8, "uint8");
for i = 1 : size(ds.Files, 1)
    img = imresize(imadjust(ds.read()), [224 224]);
    img = cat(3, img, img,img);
    hexagonals(:, :, :, (i-1)*8 + 1) = img;
    hexagonals(:, :, :, (i-1)*8 + 2) = imrotate(img, 90);
    hexagonals(:, :, :, (i-1)*8 + 3) = imrotate(img, 180);
    hexagonals(:, :, :, (i-1)*8 + 4) = imrotate(img, 270);
    hexagonals(:, :, :, (i-1)*8 + 5) = flip(img, 1);
    hexagonals(:, :, :, (i-1)*8 + 6) = imrotate(flip(img, 1), 90);
    hexagonals(:, :, :, (i-1)*8 + 7) = imrotate(flip(img, 1), 180);
    hexagonals(:, :, :, (i-1)*8 + 8) = imrotate(flip(img, 1), 270);
end

ds = imageDatastore('square');
squares = zeros(224, 224, 3, 83*8, "uint8");
for i = 1 : size(ds.Files, 1)
    img = imresize(imadjust(ds.read()), [224 224]);
    img = cat(3, img, img,img);
    squares(:, :, :, (i-1)*8 + 1) = img;
    squares(:, :, :, (i-1)*8 + 2) = imrotate(img, 90);
    squares(:, :, :, (i-1)*8 + 3) = imrotate(img, 180);
    squares(:, :, :, (i-1)*8 + 4) = imrotate(img, 270);
    squares(:, :, :, (i-1)*8 + 5) = flip(img, 1);
    squares(:, :, :, (i-1)*8 + 6) = imrotate(flip(img, 1), 90);
    squares(:, :, :, (i-1)*8 + 7) = imrotate(flip(img, 1), 180);
    squares(:, :, :, (i-1)*8 + 8) = imrotate(flip(img, 1), 270);
end

ds = imageDatastore('triangle');
triangles = zeros(224, 224, 3, 73*8, "uint8");
for i = 1 : size(ds.Files, 1)
    img = imresize(imadjust(ds.read()), [224 224]);
    img = cat(3, img, img,img);
    triangles(:, :, :, (i-1)*8 + 1) = img;
    triangles(:, :, :, (i-1)*8 + 2) = imrotate(img, 90);
    triangles(:, :, :, (i-1)*8 + 3) = imrotate(img, 180);
    triangles(:, :, :, (i-1)*8 + 4) = imrotate(img, 270);
    triangles(:, :, :, (i-1)*8 + 5) = flip(img, 1);
    triangles(:, :, :, (i-1)*8 + 6) = imrotate(flip(img, 1), 90);
    triangles(:, :, :, (i-1)*8 + 7) = imrotate(flip(img, 1), 180);
    triangles(:, :, :, (i-1)*8 + 8) = imrotate(flip(img, 1), 270);
end
%%
trainImages = zeros(224, 224, 3, 461*8, "uint8");
trainImages(:, :, :, 1:97*8) = circles;
trainImages(:, :, :, 777:197*8) = crosses;
trainImages(:, :, :, 1577:2440) = hexagonals;
trainImages(:, :, :, 2441:3104) = squares;
trainImages(:, :, :, 3105:461*8) = triangles;
trainImages = 255 - trainImages;
trainGt = categorical([repmat("circle", 97*8, 1); repmat("cross", 100*8, 1); repmat("hexagonal", 108*8, 1); repmat("square", 83*8, 1); repmat("triangle", 73*8, 1);]); 
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
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.7, ...
    'SquaredGradientDecayFactor', 0.99, ...
    'L2Regularization', 0.0001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', true, ...
    'ValidationData', {validImages, validGt}, ...
    'ValidationPatience', 50, ...
    'ExecutionEnvironment', 'gpu' ...
    );

net = trainNetwork(trainImages, trainGt, lgraph, options);