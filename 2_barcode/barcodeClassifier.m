%% Initialization
clear all
close all
clc

load('colorNet');
%% Hyper-parameter
close all
areaThreshold = 80000
levelThreshold = 0.90;%0.83;
%levelThreshold = 0.7;

data = imageDatastore('test', 'FileExtensions', '.tif');


[I, nm] = data.read();
I = I(:, :, 1:3);
mask = not(im2bw(I, levelThreshold));
imwrite(mask, strcat("temp/mask.bmp"));
nm

imshow(I)
hold on
stat = regionprops(mask, "BoundingBox");

cnt = 0;
for i = 1 : numel(stat)
    bb = stat(i).BoundingBox;
    area = bb(3) * bb(4);
    if and(area > areaThreshold/2, area < areaThreshold * 2)
        best = bb;
        %area

        candidate = imcrop(mask, best);
        ratioBest = abs(3- (bb(4) / bb(3)));
        angleBest = 0;
        bestRotated = [0, 0, size(candidate, 2), size(candidate, 1)];
        for deg = 1:1:180
            rotated = imrotate(candidate, deg);
            statRotated = regionprops(rotated, "BoundingBox");
            for j = 1 : numel(statRotated)
                bbRotated = statRotated(j).BoundingBox;
                areaRotated = bbRotated(3) * bbRotated(4);
                if and(areaRotated > areaThreshold/2, areaRotated < areaThreshold * 2)
                    %if and((bbRotated(3) / bbRotated(4) > 2), (bbRotated(3) / bbRotated(4) < 4))
                        if ratioBest < abs(3 - (bbRotated(4) / bbRotated(3)))
                            ratioBest = abs(3 - (bbRotated(4) / bbRotated(3)));
                            angleBest = deg;
                            bestRotated = bbRotated;
                        end
                    %end
                end
            end
        end
        candidate = imcrop(imrotate(imcrop(I, best), angleBest), bestRotated);
        
        sz = size(candidate);
        sample = candidate(:, floor(sz(2) / 3):floor(sz(2) / 3) * 2, :);
        %sample = imresize(sample, [224, 224]);
        offsetH = 112;
        offsetV = 45;
        sample = imresize(sample, [224 + offsetV*2, 224 + offsetH*2]);
        sample = sample(1+offsetV:224+offsetV, 1+offsetH:224+offsetH, :);
        result = colorNet.classify(sample);
        colorNet.predict(sample)
        
        if string(result) == 'blue'
            rectangle('position', best, 'EdgeColor', 'b', 'LineWidth', 4);
            text(bb(1), bb(2)-35, string(result), "FontSize",20, "FontWeight","bold");
        elseif string(result) == 'yellow'
            rectangle('position', best, 'EdgeColor', 'y', 'LineWidth', 4);
            text(bb(1), bb(2)+bb(4)+35, string(result), "FontSize",20, "FontWeight","bold");
        elseif string(result) == 'red'
            rectangle('position', best, 'EdgeColor', 'r', 'LineWidth', 4);
            text(bb(1)+bb(3)+10, bb(2)+10, string(result), "FontSize", 20, "FontWeight","bold");
        end

        % cnt = cnt + 1;
        

        %text(bb(1), bb(2)-35, string(result), "FontSize",20, "FontWeight","bold");
        imwrite(candidate, strcat("temp/", string(cnt), ".bmp"));
        imwrite(sample, strcat("temp/", string(cnt), "_sample.bmp"));
    end
end