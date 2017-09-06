function f=feature_extractor(st,a,mdp,selected_features)

if not(exist('selected_features','var'))
    selected_features=1:5;
end
 

% stde=@(st) st(1)*st(2)/((st(1)+st(2))^2+(st(1)+st(2)+1));
% t = st(1)+ st(2);
% b = max(st(1),st(2))/t;
% mvoc = st(1)/t*max(st(1)+1,st(2))/(t+1) + st(2)/t*max(st(1),st(2)+1)/(t+1);
% vpi = valueOfPerfectInformationBernoulli(st(1),st(2));
% 
% if a == 2
%     f = [stde(st), -t*mdp.cost, 0, mvoc-b, b*mdp.rewardCorrect, 1];
% else
%     f = [stde(st), -t*mdp.cost, mdp.cost, mvoc-b, mvoc*mdp.rewardCorrect - mdp.cost, 1];
% end
% 
% f = f';

if ismember(1,selected_features)
    vpi = valueOfPerfectInformationBernoulli(st(1),st(2),mdp.rewardCorrect,mdp.rewardIncorrect);
else
    vpi=NaN;
end

if ismember(2,selected_features)
    t = st(1)+ st(2);
    mvoi = 1/(t*(t+1))*(st(1)*(max(st(1)+1,st(2))-min(st(1)+1,st(2))) + st(2)*(max(st(1),st(2)+1)-min(st(1),st(2)+1)));
    voi1 = mvoi-max(st)/sum(st)-mdp.cost+min(st)/sum(st);
else
    voi1=NaN;
end

if ismember(3,selected_features)
    voc2 = nvoc(3,st,mdp.cost);
else
    voc2=NaN;
end

if ismember(5,selected_features)
    cost = mdp.cost;
else
    voc2=NaN;
end

if a == 1
    f = [vpi, voi1, voc2, max(st)/sum(st)-min(st)/sum(st), cost, 1]';
elseif a == 2
    f = [0,0,0,max(st)/sum(st)-min(st)/sum(st),0,1]';
end

f = f(selected_features,1);

end