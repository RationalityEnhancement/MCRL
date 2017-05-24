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

if ismember(2,selected_features)
    t = st(1)+ st(2);
    mvoc = 1/(t*(t+1))*(st(1)*max(st(1)+1,st(2)) + st(2)*max(st(1),st(2)+1));
    voc1 = mvoc-max(st)/sum(st)-mdp.cost;
else
    voc1=NaN;
end

if ismember(3,selected_features)
    voc2 = nvoc(3,st,mdp.cost);
else
    voc2=NaN;
end

if ismember(1,selected_features)
    vpi = valueOfPerfectInformationBernoulli(st(1),st(2));
else
    vpi=NaN;
end

if a == 1
    f = [vpi, voc1, voc2,max(st)/sum(st), 1]';
elseif a == 2
    f = [0,0,0,max(st)/sum(st),1]';
end

f = f(selected_features,1);

end