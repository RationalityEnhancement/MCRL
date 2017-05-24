function is_equal=structsEqual(struct1,struct2)

[same,dif1,dif2]=comp_struct(struct1,struct2);

is_equal=and(isempty(dif1),isempty(dif2));

end