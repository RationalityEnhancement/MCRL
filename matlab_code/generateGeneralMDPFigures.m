fig=figure(),
er = method_ers;
csterr = 1.96*method_sterrs/sqrt(3000);
errorbar(costs(1:numel(costs)),er(1:numel(costs),1),csterr(1:numel(costs),1),'LineWidth',3),hold on
errorbar(costs(1:numel(costs)),er(1:numel(costs),2),csterr(1:numel(costs),2),'LineWidth',3),hold on
errorbar(costs(1:numel(costs)),er(1:numel(costs),3),csterr(1:numel(costs),3),'LineWidth',3),hold on
errorbar(costs(1:numel(costs)),er(1:numel(costs),4),csterr(1:numel(costs),4),'LineWidth',3),hold on
errorbar(costs(1:numel(costs)),er(1:numel(costs),5),csterr(1:numel(costs),5),'LineWidth',3)

set(gca,'FontSize',16,'XScale','log')
xlabel('Cost','FontSize',18)
ylabel('Expected Return (95% Confidence Intervals)','FontSize',18)
legend('Optimal','Linear Fit','BSARSA','Meta-Greedy Policy','BO','location','Southwest')
saveas(fig,'../results/figures/GeneralMDPMethods.fig')
saveas(fig,'../results/figures/GeneralMDPMethods.png')