function [val,subscripts]=min2d(A)


[min_values,min_subscripts]=min(A);

[val,subscripts(2)]=min(min_values);
subscripts(1)=min_subscripts(subscripts(2));


end