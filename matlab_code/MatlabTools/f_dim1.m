function fxi = f_dim1(f,x,d,x_d)
    %create 1-dimensional version of a multi-dimensional function
    %the chosen dimension is d
    %and the value on that dimension is x_d
    xx    = x;
    xx(d) = x_d;
    fxi   = f(xx);
end