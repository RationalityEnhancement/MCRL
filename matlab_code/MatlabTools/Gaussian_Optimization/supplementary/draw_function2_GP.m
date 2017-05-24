function GPplot99 = draw_function2_GP(x_min,x_max,f_GP,x_scale,GP_c)

hold on

x_max_1 = [x_max 1] * x_scale(:,1);
x_max_2 = [x_max 1] * x_scale(:,2);

x_min_1 = [x_min 1] * x_scale(:,1);
x_min_2 = [x_min 1] * x_scale(:,2); 

step_size_1 = (x_max-x_min) * 0.05 * x_scale(:,1);
step_size_2 = (x_max-x_min) * 0.05 * x_scale(:,2);

xseries1 = x_min_1:step_size_1:x_max_1;
xseries2 = x_min_2:step_size_2:x_max_2;

for i=1:size(xseries1,2)
  for j=1:size(xseries2,2)
    [m s2] = f_GP([xseries1(i) xseries2(j)]);
    yseries(i,j) = m+GP_c*sqrt(s2);
  end
end

%surf(xseries1,xseries2, yseries);
GPplot99 = mesh(xseries1,xseries2, yseries);
colormap gray



%axis([0 1 -0.3 1.3]);


end