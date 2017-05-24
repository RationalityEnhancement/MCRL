function fxi = f_dim1(f,x,i,xi)
    xx    = x;
    xx(i) = xi;
    fxi   = f(xx);
end