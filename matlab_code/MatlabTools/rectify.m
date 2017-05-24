function x_rect=rectify(x,x_min,x_max)
    x_rect=min(x_max,max(x_min,x));
end
