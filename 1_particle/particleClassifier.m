%% Initialization
clear all
close all
clc

load('shapeNet'); % load shape classifier
load('colorNet'); % load color classifier

Font = 30; % visualization font
%% Hyper-parameter
magnification = 10; % [10, 20], 10x lens or 20x lens of microscope

areaThreshold = -1;
levelThreshold = 0.83;
ratioThreshold = 0.3;
classThreshold = 0.90;
colorThreshold = 0.90; % 0.96;

if magnification == 10 % 10x lens
    areaThreshold = 14 * 10000;
elseif magnification == 20 % 20x lens
    areaThreshold = 70 * 10000;
end
%% Read target meta data
red = imageDatastore('data/red_cross','FileExtensions', '.tif');
blue = imageDatastore('data/blue_hexagonal','FileExtensions', '.tif');
yellow = imageDatastore('data/yellow_circle','FileExtensions', '.tif');
test = imageDatastore('data/test', 'FileExtensions', '.tif');
%% Computation
[I, nm] = test.read(); % red.read(), blue.read(), yellow.read(), test.read()
mask = not(im2bw(I, levelThreshold)); % image to binary image with level threshold
nm % file description
imshow(I)
hold on
stat = regionprops(mask, "BoundingBox"); % Extract image features
saad = zeros(224, 224, 3, 11, 'uint8'); cnt = 1;
for i = 1 : numel(stat)
    bb = stat(i).BoundingBox;
    area = bb(3) * bb(4);
    ratio = bb(3) / bb(4);
    if and(area > areaThreshold, area < areaThreshold * 4)
        if and(ratio > 1 - ratioThreshold, ratio < 1 + ratioThreshold)
            best = bb;
            
            x = floor(bb(1))+1; % start x coordinate of boundingbox
            y = floor(bb(2))+1; % start y coordinate of boundingbox
            xDestination = x + floor(bb(3)) - 1; % destination x coordinate of boundingbox
            yDestination = y + floor(bb(4)) - 1; % destination y coordinate of boundingbox
            cx = floor((x + xDestination) / 2); % center x
            cy = floor((y + yDestination) / 2); % center y
            lx = floor(bb(3) / 6); % sampling x range
            ly = floor(bb(4) / 6); % sampling y range
            
            % cropping image and mask
            croppedColor = I(cy-ly:cy+ly, cx-lx:cx+lx, :);
            croppedMask = rgb2gray(I(y:yDestination, x:xDestination, :));

            % transformation
            %croppedMaskUint8 = uint8(croppedMask);
            croppedMaskResize = imadjust(imresize(croppedMask, [224, 224]));
            croppedStack = zeros(224, 224, 3, "uint8");
            croppedStack(:, :, 1) = croppedMaskResize;
            croppedStack(:, :, 2) = croppedMaskResize;
            croppedStack(:, :, 3) = croppedMaskResize;
            %imwrite(rgb2gray(croppedStack), strcat(string(zzzz), '_', string(i), '.bmp'));
            croppedStack = 255 - croppedStack;
            
            % shape recognition
            classPrediction = net.predict(croppedStack);
            classPrediction = max(classPrediction);
            class = net.classify(croppedStack);
            
            % color recognition
            colorIn = imresize(I(y:yDestination, x:xDestination, :), [224, 224]);
            colorPrediction = colorNet.predict(colorIn);
            colorPrediction = max(colorPrediction);
            stringColor = colorNet.classify(colorIn);
            saad(:, :, :, cnt) = croppedStack; cnt = cnt + 1;
            if classPrediction > classThreshold
                if colorPrediction > colorThreshold
                    if and(string(class) == 'circle', stringColor == "yellow")
                        rectangle('position',best,'edgecolor','g','linewidth',4);
                        text(bb(1), bb(2)-280, string(class), "FontSize",Font, "FontWeight", "bold")
                        text(bb(1), bb(2)-140, stringColor, "FontSize",Font, "FontWeight", "bold")
                    elseif and(string(class) == 'hexagonal', stringColor == "blue")
                        rectangle('position',best,'edgecolor','g','linewidth',4);
                        text(bb(1), bb(2)-280, string(class), "FontSize",Font, "FontWeight", "bold")
                        text(bb(1), bb(2)-140, stringColor, "FontSize",Font, "FontWeight", "bold")
                    elseif and(string(class) == 'cross', stringColor == "red")
                        rectangle('position',best,'edgecolor','g','linewidth',4);
                        text(bb(1), bb(2)-280, string(class), "FontSize",Font, "FontWeight", "bold")
                        text(bb(1), bb(2)-140, stringColor, "FontSize",Font, "FontWeight", "bold")
                    elseif (string(class) == 'triangle') % 임시
                        rectangle('position',best,'edgecolor','r','linewidth',4);
                        text(bb(1)-740, bb(2)+70, string(class), "FontSize",Font, "FontWeight", "bold")
                        text(bb(1)-740, bb(2)+210, stringColor, "FontSize",Font, "FontWeight", "bold")
                    else
                        rectangle('position',best,'edgecolor','r','linewidth',4);
                        text(bb(1), bb(2)-280, string(class), "FontSize",Font, "FontWeight", "bold")
                        text(bb(1), bb(2)-140, stringColor, "FontSize",Font, "FontWeight", "bold")
                    end
                    %rectangle('position', [xDestination-100, y-110, 100, 100], 'FaceColor', [r, g, b]/255, 'EdgeColor', [0, 0, 0])
                else
                    rectangle('position',best,'edgecolor','r','linewidth',4);
                    text(bb(1), bb(2)-280, "ambiguous", "FontSize", Font, "FontWeight", "bold")
                    text(bb(1), bb(2)-140, "color", "FontSize", Font, "FontWeight", "bold")
                end
            else
                rectangle('position',best,'edgecolor','r','linewidth',4);
                text(bb(1), bb(2)-280, "unknown", "FontSize", Font, "FontWeight", "bold")
                text(bb(1), bb(2)-140, "shape", "FontSize", Font, "FontWeight", "bold")
            end
        else
            best = bb;
            %area

            rectangle('position',best,'edgecolor','r','linewidth',4);
            text(bb(1), bb(2)-140, "out of ratio", "FontSize", Font, "FontWeight", "bold")
        end
    end
end

