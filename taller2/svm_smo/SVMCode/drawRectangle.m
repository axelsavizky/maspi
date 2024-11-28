function drawRectangle(box,color,lw)
% fonction drawRectangle(image,box -> [ x y dw dh], nro_boxes)
nro_box = size(box,1);
hold on;
for j=1:nro_box
    plot([box(j,1) box(j,1)+box(j,3) box(j,1)+box(j,3) box(j,1) box(j,1)],...
    [box(j,2) box(j,2) box(j,2)+box(j,4) box(j,2)+box(j,4) box(j,2)],'-','Color',color,'LineWidth',lw);
end
hold off;
