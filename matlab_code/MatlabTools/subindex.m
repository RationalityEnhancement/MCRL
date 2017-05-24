function val = subindex(array, varargin)

subs=varargin;
subs(end+1:ndims(array))={':'};
val = subsref(array, struct('type','()', 'subs', {subs}));

end