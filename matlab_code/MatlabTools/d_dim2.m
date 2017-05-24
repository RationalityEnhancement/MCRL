function fxixj = f_dim2(f,x,d1,d2,x_d1d2)
    %create 2-dimensional version of a multi-dimensional function
    %the chosen dimensions are d1 and d2
    %and the argument value on those dimension is x_d1d2
    xx    = x;
    xx(d1) = x_d1d2(1);
    xx(d2) = x_d1d2(2);
    fxixj = f(xx);
end
