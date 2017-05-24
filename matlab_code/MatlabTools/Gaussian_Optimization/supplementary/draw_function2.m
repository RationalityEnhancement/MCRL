function draw_function2(x_min,x_max,f,x_scale)

cla
hold on

x_max_1 = [x_max 1] * x_scale(:,1);
x_max_2 = [x_max 1] * x_scale(:,2);

x_min_1 = [x_min 1] * x_scale(:,1);
x_min_2 = [x_min 1] * x_scale(:,2); 

step_size_1 = (x_max-x_min) * 0.01 * x_scale(:,1);
step_size_2 = (x_max-x_min) * 0.01 * x_scale(:,2);

xseries1 = x_min_1:step_size_1:x_max_1;
xseries2 = x_min_2:step_size_2:x_max_2;

for i=1:size(xseries1,2)
  for j=1:size(xseries2,2)
    yseries(i,j) = f([xseries1(i) xseries2(j)]);
  end
end

%surf(xseries1,xseries2, yseries);
%colormap white 

mesh(xseries1,xseries2, yseries);
colormap gray
%colormap jet

%axis([0 1 -0.3 1.3]);


end