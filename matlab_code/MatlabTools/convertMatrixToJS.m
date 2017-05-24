function js_matrix_str=convertMatrixToJS(matrix,var_name)

if not(exist('var_name','var'))
    var_name=inputname(1);
end

js_matrix_str=[var_name,'=new Array(); '];

nr_dimensions=ndims(matrix);
nr_elements=size(matrix);

if nr_dimensions==1
    matrix_str=strrep(mat2str(matrix(:)),';',',');
    js_matrix_str=[var_name,' = ',matrix_str];
elseif nr_dimensions==2
    for r=1:nr_elements(1)
        row_str=strrep(mat2str(squeeze(matrix(r,:))'),';',',');
        js_matrix_str=[js_matrix_str,' ',var_name,'.push(',row_str,');'];
    end    
elseif nr_dimensions==3
    for r=1:nr_elements(1)
        temp=squeeze(matrix(r,:,:));
        js_matrix_str=[js_matrix_str,convertMatrixToJS(temp),';',var_name,'.push(temp);'];
        %{
        js_matrix_str=[js_matrix_str,'temp=new Array();'];
        for c=1:nr_elements(2)                        
            col_str=strrep(mat2str(squeeze(matrix(r,c,:))),';',',');            
            js_matrix_str=[js_matrix_str,' temp.push(',col_str,');'];
        end
        js_matrix_str=[js_matrix_str,' ',var_name,'.push(temp);'];
        %}
    end    
        
else
    throw(MException('MException','The input matrix has too many dimensions.'))
end

end