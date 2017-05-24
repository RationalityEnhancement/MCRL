function rounded=roundToMultipleOf(x,divisor)
%Rounds the first input argument to the closest multiple of the second
%argument.

rounded=round(x/divisor)*divisor;

end