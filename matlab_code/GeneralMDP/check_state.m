% Get the state-action pair using the X-value in VOC vs VOC_hat plot
in = -0.1918;
[A,c] = min(abs(voc_hat - in));
[i,j] = find(state_action == c)

%Get the specific state and calculate features
st = S(i,:);  
st_m = reshape(st,2,nr_arms)';
er = max(st_m(:,1) ./ sum(st_m,2));

vpiij = valueOfPerfectInformationMultiArmBernoulli(st_m(:,1),st_m(:,2),j);
voc1ij = VOC1MultiArmBernoulli(st_m(:,1),st_m(:,2),j,cost)-er;

% Check Results
f = [voc1ij,vpiij,er,1]
f*w
fp = X(c,:) %check features
fp*w %check evaluation 
vocl(c) %check y-value