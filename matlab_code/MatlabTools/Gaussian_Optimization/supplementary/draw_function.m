function draw_function(x_min,x_max,f,clrblue,x_scale)

%cla
hold on

x_max = [x_max 1] * x_scale(:,1);
x_min = [x_min 1] * x_scale(:,1);
step_size = (x_max-x_min) * 0.01 * x_scale(:,1);

xseries = x_min:step_size:x_max;
yseries = arrayfun(f,xseries);

%plot(xseries,yseries,'Linewidth',1,'Color',[0 0 clrblue]);

plot(xseries,yseries,'Linewidth',1,'Color',[0 0 0]);


%axis([0 1 -0.3 1.3]);


end