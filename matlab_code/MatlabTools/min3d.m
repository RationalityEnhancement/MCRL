function [val,subscripts]=min3d(A)


[min_valuesd1,min_subscriptsd1]=min(A);

[min_valuesd2,min_subscriptsd2]=min(min_valuesd1);

[min_valuesd3,min_subscriptsd3]=min(min_valuesd2);

subscripts(3)=min_subscriptsd3;
subscripts(2)=min_subscriptsd2(1,1,subscripts(3));
subscripts(1)=min_subscriptsd1(1,subscripts(2),subscripts(3));

val=min_valuesd3;


end