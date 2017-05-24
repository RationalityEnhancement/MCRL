function eq=modelEquation(variable_names,w)

nr_variables=numel(variable_names);

eq=[num2str(w(1)),'*',variable_names{1}];
for v=2:nr_variables
    
    if w(v)>0
        eq=[eq,'+',num2str(w(v)),'*',variable_names{v}];
    elseif w(v)<0
        eq=[eq,num2str(w(v)),'*',variable_names{v}];
    end
end

end