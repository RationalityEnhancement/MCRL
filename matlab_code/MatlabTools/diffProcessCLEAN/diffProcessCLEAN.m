%SIMULATION OF THE DIFFUSION PROCESS -  VARIABILITY ACROSS TRIALS, plus
%other types of variability (variability of nondecision time, variability
%of starting point)
%this model will work with one or two barriers. If you want one barrier,
%just put b=-inf;
%-----------------
%THIS IS A GENERAL PURPOSEFULLY MODEL. It can be used with different kind
%of diffusion process (one/two boundaries, variability across trials,
%other kind of variabilitis).
%It is used for generating data. No plot showed during the process.

%CLEAN:This is a clean version for the blog/MATLAB file exchange.
%https://biscionevalerio.wordpress.com

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   INPUT PARAMETERS: see description below
%   OUTPUT: resp is a matrix 2x2, the first column contains
%   the RT (in seconds), the second contains 1 for correct (upper
%   threshold), 0 for incorrect (lower threshold). RT is NAN is the process
%   did not terminate.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Valerio Biscione.
%18/12/2014.
function [resp]=diffProcess(varargin)
p=inputParser;
addParamValue(p, 'numTr', 1500); %number of trials (iteration of the simulation)
addParamValue(p, 'a',0.16 ); %upper threshold
addParamValue(p, 'b', []); %lower threshold. If []. b=0
addParamValue(p, 'z', []); %starting point, if [] then z=a/2

addParamValue(p, 'v', 0.3); %drift rate within trial
addParamValue(p, 'Ter', .26);  %Non decision time
addParamValue(p, 'st', 0); %variability in the non decision time
addParamValue(p, 'eta', 0.063);  %variability in drift rate across trial
addParamValue(p, 'sz', 0);  %variability in starting point
addParamValue(p, 'c', 0.1); %std within trial, put [] is you want it to calculate for you
addParamValue(p, 'tau', 0.0001); %step
%notice that s
addParamValue(p, 'maxWalk', 2500); %the number of points for each trunck of cumsum
parse(p,varargin{:}); numTrials=p.Results.numTr; a=p.Results.a; b=p.Results.b; z=p.Results.z;
%s=p.Results.s;
v=p.Results.v;  Ter=p.Results.Ter; st=p.Results.st; eta=p.Results.eta; sz=p.Results.sz; tau=p.Results.tau; c=p.Results.c;
maxWalk=p.Results.maxWalk;   resp=zeros(numTrials,2);


%if plotF==1,h=figure(); subplot(3,2,[3,4]);  box on; hold on;set(h, 'Position', [-1400 -50 1400 800]); end

if isempty(z),    z=a/2;end


mu=a/v; sig=(a.^2./(c^2));

for xx=1:numTrials
    timeseries(1:maxWalk,1)=NaN;
    zz=unifrnd(z-sz/2,z+sz/2,1,1); %real starting point for this trial;    
    upB=a-zz;
    if isempty(b),lowB=-zz; else lowB=b-zz; end 
    %we uniform everything such that the starting point is always 0. 
    zz=0;
    startPoint=zz;
    vm=normrnd(v,eta,1,1);
    % vm=unifrnd(v-eta/2,v+eta/2,1,1); %maybe you want to use a different
    % drift rate distribution?
    
    index=1;
    for ii=1:100
        timeseriesSUM=cumsum([startPoint; normrnd(vm*tau,c*sqrt(tau),maxWalk,1)]);
        firstPassA=find(timeseriesSUM(2:end)>=upB,1); firstPassB=find(timeseriesSUM(2:end)<=lowB,1);
        i=min([firstPassA firstPassB]);
        if isempty(i)
            startPoint(1)=timeseriesSUM(end);
            continue;
        else
            index=i+1+((ii-1)*maxWalk);
            break;
        end
    end
    
    %if the process DO NOT terminate
    if isempty(firstPassB) && isempty(firstPassA)
        resp(xx,1)=-1;
        resp(xx,2)=3;
    else
        %if it DOES terminate, we want to now which was the first passage
        %point.
        resp(xx,1)=(index*tau)+unifrnd(Ter-st/2,Ter+st/2,1,1); %CORRECT
        if isempty(firstPassB) && ~isempty(firstPassA)
            resp(xx,2)=1;
        elseif ~isempty(firstPassB) && isempty(firstPassA)
            resp(xx,2)=0;
        elseif (~isempty(firstPassB) && ~isempty(firstPassA) && firstPassA<firstPassB)
            resp(xx,2)=1;
        elseif (~isempty(firstPassB) && ~isempty(firstPassA) && firstPassA>firstPassB)
            resp(xx,2)=0;
        else
        end
    end
end


%rt =0 means that it reached the deadline without response. We set this
%values to NaN.
resp(resp(:,1)==0,:)=NaN;
end
%}