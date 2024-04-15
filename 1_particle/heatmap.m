croppedStack = saad(:, :, :, 11);
scoreMap = gradCAM(net, croppedStack, "hexagonal");
figure
imshow(croppedStack)
hold on
imagesc(scoreMap,'AlphaData',0.5)
colormap jet
colorbar