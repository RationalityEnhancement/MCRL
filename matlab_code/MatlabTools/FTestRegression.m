function [F,df_N,df_D,p]=FTestRegression(y,X,var_number,options)

nr_predictors=size(X,2);
if exist('options','var')
    glm_full=fitglm(X,y,options{1},options{2});
    glm_restricted=fitglm(X(:,setdiff(1:nr_predictors,var_number)),y,options{1},options{2});
else
    glm_full=fitglm(X,y);
    glm_restricted=fitglm(X(:,setdiff(1:nr_predictors,var_number)),y);    
end

F=(glm_restricted.SSE-glm_full.SSE)/(glm_restricted.DFE-glm_full.DFE)/...
    (glm_full.SSE/glm_full.DFE);

df_N=glm_restricted.DFE-glm_full.DFE;
df_D=glm_full.DFE;
p=1-fcdf(F,df_N,df_D);

end