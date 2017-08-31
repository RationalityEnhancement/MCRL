function f = getFeatures(st,c)
    %the below line can be commented out to improve
    %performance if the results are already loaded
    %in the environment
    load ../results/lightbulb_problem.mat 
    if nargin == 1  
        c = 7;
    end
    X = lightbulb_problem(c).fit.features;
    S = lightbulb_problem(c).mdp.states;
    I = S(:, 1) == st(1) & S(:, 2) == st(2);
    f = [X(I,1:3), (max(st)-min(st))/sum(st)];
end