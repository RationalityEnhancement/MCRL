st = [11,1];
c = 3;
co = costs(c);
X = lightbulb_problem(c).fit.features;
tic
for i=1:300
    I = find(S(:, 1) == st(1) & S(:, 2) == st(2));
    vpi1 = X(I,1);
    voc1 = X(I,2);
    
    t = sum(st);
    mvoc = 1/(t*(t+1))*(st(1)*(max(st(1)+1,st(2))-min(st(1)+1,st(2))) + st(2)*(max(st(1),st(2)+1)-min(st(1),st(2)+1)));
    voc1a = mvoc - max(st)/t - co + min(st)/t;

    if voc1a ~= voc1
        disp('fuck')
    end
end
toc