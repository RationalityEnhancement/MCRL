fig=figure();
er = results.er;
csterr = 1.96*results.sterr/sqrt(3000);
errorbar(costs(1:13),er(1:13,1),csterr(1:13,1),'LineWidth',3),hold on
errorbar(costs(1:13),er(1:13,2),csterr(1:13,2),'LineWidth',3),hold on
errorbar(costs(1:13),er(1:13,3),csterr(1:13,3),'LineWidth',3),hold on
errorbar(costs(1:13),er(1:13,4),csterr(1:13,4),'LineWidth',3),hold on
errorbar(costs(1:13),er(1:13,5),csterr(1:13,5),'LineWidth',3)

set(gca,'FontSize',16,'XScale','log')
xlabel('Cost','FontSize',18)
ylabel('Expected Return (95% Confidence Intervals)','FontSize',18)
legend('Optimal','BSARSA','Linear Fit','Meta-Greedy Policy','BO','location','Southwest')
saveas(fig,'../results/figures/MetaMDPMethods.fig')
saveas(fig,'../results/figures/MetaMDPMethods.png')