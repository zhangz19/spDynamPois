function [] = spDynamPois_demo()
rng('default');  rng(8);  %set random seed

%Y: N by T spatio-temporal count data
%E: N by T offset
%W: N by N spatial adjacency matrix (binary)
%X: N by p by T: p-covariates. X(:,:,t) should match with Y(:,t) at time t
load('spDynamPois_demo_data.mat','Y','X','E','W')

% model setting: set environment variables (ev) 
ev.verbose = true; 
ev.nonspat = 0; %1=spatial (CAR), 0=nonspatial
% ev.niter = 2e4;  ev.burnin = 15e3;  %MCMC sample = 2e4 - 15e3
ev.niter = 50;  ev.burnin = 0;  % short MCMC run for this demo
ev.adaptMCMC = true; 
ev.useMetropolis_eta = true; %if true: metropolis for updating eta, false then use griddy sampler
ev.beta0prior = 1; %1=flat, 2=normal
ev.crossValidation = false;   ev.crossValidationPercent = 0.1;  % cross validation

out = spDynamPois(Y, X, E, W, ev); 

% posterior summary of parameters
summaryPara = prctile(out.matPara,[2.5 25 50 75 97.5]);  %Percentiles
disp(summaryPara(:,1:(size(X,2)*size(X,3))))  % summarize for beta(j,t), jth covariate effect at time t

%To learn more about the demo data or cite for the data/model: 
% @article{snow2018regional,
%   title={Regional-based mitigation to reduce wildlife--vehicle collisions},
%   author={Snow, Nathan P and Zhang, Zhen and Finley, Andrew O and Rudolph, Brent A and Porter, William F and Williams, David M and Winterstein, Scott R},
%   journal={The Journal of Wildlife Management},
%   volume={82},
%   number={4},
%   pages={756--765},
%   year={2018},
%   publisher={Wiley Online Library}
% }

end
