%% Generate Figure
clear
load('../results/BO_validation')
load('../results/greedy.mat')
load('../results/blinkered.mat')
load('../results/BSARSA.mat')
load('../results/full_observation.mat')
load('../results/no_deliberation.mat')

[costs_BO_sorted,order]=sort(BO_validation.cost);

fig=figure(),
errorbar(costs_BO_sorted',BO_validation.ER(order),BO_validation.sem(order),'LineWidth',3),hold on
errorbar(BSARSA.cost,BSARSA.ER,BSARSA.sem_ER,'LineWidth',3),hold on
errorbar(full_observation.cost,full_observation.ER,full_observation.sem_ER,'LineWidth',3),hold on
errorbar(greedy.cost,greedy.ER,greedy.sem_ER,'LineWidth',3),hold on
errorbar(no_deliberation.cost,no_deliberation.ER,no_deliberation.sem_ER,'LineWidth',3),hold on
errorbar(blinkered.costs,blinkered.ER,blinkered.sem_ER,'LineWidth',3)
xlim([0.01,13])
ylim([0,40])
set(gca,'FontSize',16,'XScale','log')
xlabel('Cost per Click','FontSize',16)
ylabel('Expected return','FontSize',16)
legend('BO','BSARSA','Full-Deliberation Policy','Meta-Greedy Policy','No-Deliberation Policy','Blinkered','location','West')
saveas(fig,'../results/figures/MouselabValidation.fig')
saveas(fig,'../results/figures/MouselabValidation.png')

for c_ind=1:numel(costs_BO)
    BO_ind=find(costs_BO==costs_BO(c_ind));
    greedy_ind=find(costs==costs_BO(c_ind));
    t_BO_vs_greedy(c_ind)=(BO_validation.ER(BO_ind)-avg_ER_greedy(greedy_ind))./sqrt(BO_validation.sem(BO_ind).^2+...
        sem_ER_greedy(greedy_int).^2);
    p(c_ind)=1-normcdf(abs(t_BO_vs_greedy(c_ind)))
end

threshold=0.05/13
[p<threshold; t_BO_vs_greedy; costs]



%test performance of BSARSA against the performance of the meta-greedy
%policy and the full-observation policy
t_BSARSA_vs_greedy=(avg_performance.BSARSAQ_validation-avg_ER_greedy)./sqrt(sem_performance.BSARSAQ_validation.^2+...
    sem_ER_greedy.^2);
p=1-normcdf(abs(t_BSARSA_vs_greedy))
threshold=0.05/13
[p<threshold; t_BSARSA_vs_greedy; costs]

t_BSARSA_vs_full_observation=(avg_performance.BSARSAQ_validation-avg_performance.full_observation)./...
    sqrt(sem_performance.BSARSAQ_validation.^2+sem_performance.full_observation.^2);
p=1-normcdf(abs(t_BSARSA_vs_full_observation))
threshold=0.05/13
[p<threshold; t_BSARSA_vs_full_observation; costs]

t_BSARSA_vs_greedy=(avg_performance.BSARSAQ_validation-avg_ER_greedy)./sqrt(sem_performance.BSARSAQ_validation.^2+...
    sem_ER_greedy.^2);
p=1-normcdf(abs(t_BSARSA_vs_greedy))
threshold=0.05/13
[p<threshold; t_BSARSA_vs_greedy; costs]