function [] = spDynamPois(ID)

global ys  W M E indT tvmat

verbose = 1;
simu = 0;
usesubset = 0;
adaptMCMC = 1; % tune proposal variance for Metropolis steps?
updateParaCAR = 1;  %update CAR parameters?
updateW1 = 1;  %update w_tj for j=1 (log risk)?
updateW = 0; % no w_tj for j =2:q
beta0prior = 1; %1=flat, 2=normal
betaTprior = 1; %1=dynamic, 2=flat
startfromtrue = 0;
startfromsy = 1;
facw = .01;  %0.3
useMetropolis_eta = 1;
reportrate = 0;
ecobeta = 1; % varying slopes across ecozones?

% tot = 100; burnin = 0;  % test run
tot = 2e4; burnin = 15e3;

ch = str2double(num2str(ID));
ch0 = ch;

nonspat = 0;
% if nstate == 2 % nonspatial case
%     nonspat = 1;
% end
if ch > 120 %10bins, 3chs, 4 models
    nonspat = 1;
    ch = ch-30;
end

nonspat = 1; 

nChain = 3; %5;
nVar = 10; %nvar indicates time here

nstate = ceil(ch/(nChain*nVar)); %nstate indicates model id
ch = ch - (nstate-1)*(nChain*nVar);
nvar = ceil(ch/nChain); % nvar indicates cv-bin id
ch = ch - (nvar-1)*nChain;
fprintf('nonsptial=%d, model = %d, bin = %d, chain = %d:\n', [nonspat, nstate,nvar,ch])

load('datAll.mat')

Y = ys(:,1); NT = size(Y,1); N = size(W,1); T = NT/N; M = diag(sum(W,1));
Y = reshape(Y, [N,T]); E = reshape(E, [N,T]);
X = [ones(NT,1), (ys(:,2:end))];   %not standardized anymore

load('tvmat.mat')
tind = 1:N; vind = tvmat(:,nvar); vind = vind(vind>0); tind(vind) = [];
Tind = ones(N,T);
Tind(vind, T) = 0; % vind, last year for validation
Tind = reshape(Tind, [numel(Tind), 1]);

if ecobeta == 1 %eco-specific slopes
    X_tmp = X(:,  (size(X,2)-4):(size(X,2)-2)); % state
    X = X(:,1:(size(X,2)-5)); % exclude the states and ecozone intercept as we will reshape the X
    % exclude the states as well because some states never appear in one of
    % the ecozone, so must be excluded if we consider varying slopes across
    % ecozones
    
    % for model comparisons
    initind = 1:24; % full covariates, 3 ecozones
    if nstate == 1
        useind = 1;
    elseif nstate == 2
        useind = 1:5;
    elseif nstate == 3
        useind = 1:6;
    elseif nstate == 4
        useind = 1:8;
    end
    X = X(:, useind);
    for j = 1:numel(useind)
        initind(useind(j) + [0,8,16]) = 1;
    end
    initind = find(initind==1);
    
    p0 = size(X,2); J0 = max(ezone); X0 = zeros(size(X,1), J0*p0);
    for j = 1:J0
        X0(ezone==j, (j-1)*p0+(1:p0)) = X(ezone==j,:);
    end
    X = X0;
    X = [X, X_tmp];
    clear X0 X_tmp
end

p = size(X,2); % number of fixed effects
q = 1; % number of random effects: q=1 means intercept only

if usesubset == 1
    N0 = N; T0 = size(Y,2);
    N = 30; NT = N*T0;
    Y = Y(1:N, :); E = E(1:N, :); W = W(1:N, 1:N); M = diag(max(sum(W,1),1));
    indT = reshape(indT, [N0, T0]); indT = indT(1:N, :); indT = reshape(indT, [numel(indT), 1]);
    X0 = zeros(NT,p);
    for j = 1:p
        a = reshape(X(:,j), [N0, T0]); a = a(1:N, :);
        X0(:,j) = reshape(a, [numel(a),1]);
    end
    X = X0;
    clear X0 N0 T0
end

% if singleyear == 1
%     Y = Y(:, nvar); E = E(:, nvar); X = X(indT==nvar,:); indT = ones(N,1);
%     T=1;
% end

Y_vind = Y(vind, T);

nam = strcat('out_', num2str(nonspat),'_', num2str(nstate),'_', ...
    num2str(nvar),'_',num2str(ch),'.mat');

fprintf('N = %d, T = %d, p = %d, q = %d\n', [N,T,p,q])

%======================================= set hyperparameters for priors
meantau2 = 0.01; vartau2 = 10^4;  alphatau = 2+meantau2^2/vartau2;
invbetatau = meantau2*(alphatau-1);
% alphatau = 0; invbetatau = 0;
% IW_nu = p+1 + 1e-3; IW_A = 1e-3*eye(p); %p+3+ 1e-10*
IW_nu = p+2; IW_A = eye(p);

if useMetropolis_eta ~= 1
    gap_eta = 1e-2; etavec = (-4):gap_eta:4; % may need to change the upper bound  5
    len_eta = length(etavec);
    loglike0_eta = zeros(len_eta,N,T);
    for i = 1:N
        for t = 1:T
            loglike0_eta(:,i,t) = Y(i,t).*etavec - E(i,t).*exp(etavec);
        end
    end
end

invM = inv(M); eigs = eig(sqrt(invM)*W*sqrt(invM));
lphi = max(1/min(eigs),-1); uphi = 1/max(eigs);
gap_phi = 1e-2; phis = (lphi+gap_phi):gap_phi:(uphi-gap_phi); len = length(phis);
lphi0 = zeros(1,len);
for i = 1:len
    lphi0(i) = 0.5*sum(log(eig(M-phis(i)*W)));
end

% set random seeds
rng('default'); rng(ch0*5);
if simu == 1
    rng('default'); rng(25);
end

% initialize: start with non-spatial model
Yobs = log((Y + 0.5*(Y==0))./E); x.beta = zeros(p,T+1); x.tau2 = 0.04*ones(q,T);
x.w = zeros(N,T,q);
x.phi = 0.9 + zeros(q,T); %x.phi(1,1) = 0;
x.Sigma_eta = 0.01*eye(p);
for t = 1:T
    Xt = X(indT==t,:);
    %x.w(:,t,1) = Yobs(:,t) - Xt*x.beta(:,t+1);
    if simu == 1
        if t == 1
            [b, ~, stats] = glmfit(Xt(:,2:end), Y(:,t), 'poisson','offset', log(E(:,t)));
            x.beta(:,t) = b;
            x.Sigma_eta = stats.covb*1e3;
        end
        x.beta(:,t+1) = mvnrnd(x.beta(:,t), x.Sigma_eta);
        % if updateW == 1
        for j = 1:q
            Sigma_w = x.tau2(j,t)*((M - x.phi(j,t)*W)\eye(N));
            L = chol(Sigma_w, 'lower');
            if t > 1
                x.w(:,t,j) = L*normrnd(0,1,[N,1]) + x.w(:,t-1,j); %mvnrnd(x.w(:,t-1,j), Sigma_w); %
            else
                x.w(:,t,j) = L*normrnd(0,1,[N,1]);  % mvnrnd(zeros(N,1), Sigma_w); %
            end
        end
        % end
        Yobs(:,t) = Xt*x.beta(:,t+1) + sum(Xt(:,1:q).*squeeze(x.w(:,t,:)),2);
    else
        x.beta(:,t+1) = (Xt'*Xt)\(Xt'*Yobs(:,t));
    end
end
% now recode x.w(:,1,:) to be the log risk
x.w(:,:,1) = Yobs;

truePara = []; trueW = [];
if simu == 1
    Y = E.*exp(Yobs); %without Poisson rn generation
    truePara =  [reshape(x.beta,[1,numel(x.beta)]), x.Sigma_eta(~~tril(x.Sigma_eta))',...
        reshape(x.tau2,[1,numel(x.tau2)]), reshape(x.phi,[1,numel(x.phi)]) ];
    trueW = reshape(x.w, [1,numel(x.w)]);
end

if startfromsy == 1
    x.beta = []; x.tau2 = x.tau2*0; x.phi = x.phi*0; x.Sigma_eta = x.Sigma_eta*0;
    for mystate = 1:1
        load(strcat('Inits',num2str(nonspat),'.mat'))
        p0 = p/3;
        x.beta = [x.beta; inits{mystate, 1}(initind,:)];
        x.tau2 = x.tau2 + inits{mystate, 3};
        x.phi = x.phi + inits{mystate, 4};
        x.Sigma_eta = inits{mystate, 2}(initind,initind);
        x.w = inits{mystate, 5};
    end
    x.Sigma_eta = blkdiag(x.Sigma_eta, .1*eye(3));
    x.beta = [x.beta; zeros(3,size(x.beta,2))];
end

% MCMC running
npara = numel(x.beta) + numel(x.phi) + numel(x.tau2) + numel(x.Sigma_eta(~~tril(x.Sigma_eta+5)));
matPara = nan((tot-burnin), npara);
Ws = nan((tot-burnin), numel(x.w(vind,T,1)));
w_accepts = zeros(N,T); objrate = 0.44;
batchLen = 50; batchNum = 0; batchTot = tot/batchLen;
w_rates = zeros(batchTot, N*T);
% beta_ls = log(0.3*abs(x.beta(:,2:end) + 10*exp(-5)));

% ============================ initial log-std for proposal ==================================
w_ls = log(facw*(abs(x.w(:,:,1)) + 1e-6)); %repmat(log(sqrt(.02*diag(cov(x.w')))), [1,T]); %
% ======================================================================================

rng('default'); rng(ch0*13);
% x.beta = x.beta + normrnd(0,1,size(x.beta)).*abs(x.beta)*0.01;
Zpart = ones(N,T);

% if adaptMCMC == 0
for t = 1:T
    Xt = X(indT==t,:);
    [b, ~, stats] = glmfit(Xt(:,2:end), Y(:,t), 'poisson','offset', log(E(:,t)));
    if startfromtrue == 0
        x.beta(:,t+1) = b; %(Xt'*Xt)\(Xt'*Yobs(:,t));
    end
    Zpart0 = exp(sum(Xt(:,1:q).*squeeze(x.w(:,t,:)),2));
    Zpart(:,t) = Zpart0;
end
% Zpartraw = Zpart;
if startfromtrue == 0
    x.beta = x.beta + normrnd(0,1,size(x.beta)).*abs(x.beta)*0.001;
end
% end

E_BETApart = zeros(N,T); %stores E*exp(X*beta)
for t = 1:T
    Et = E(:,t); Xt = X(indT==t,:);
    Delta = Xt*x.beta(:,t+1); E_BETApart(:,t) = Et.*exp(Delta);
end
% subplot(1,2,1), plot(Y); subplot(1,2,2), plot(E_BETApart)

checkpoint = 1;
iter0 = 0; t0 = 0; completed = 0;
if exist(nam, 'file')
    load(nam)
    if completed == 1
        error('completed already')
    end
end

if nonspat == 1 % nonspatial case
    x.phi = x.phi*0; M = eye(size(M,1));
end

tic
for iter = (iter0+1):tot
    if verbose==1; fprintf('%6d', iter); if(~mod(iter,20)); fprintf('\n'); end; end
    
    %+++++++++++++++  construct CAR covariance matrices
    Dt = cell(q,T);
    for t = 1:T
        for j = 1:q
            if t ~= T; Dt{j,t} = (M - x.phi(j,t)*W)/x.tau2(j,t);
            else Dt{j,t} = (M(tind, tind) - x.phi(j,t)*W(tind, tind))/x.tau2(j,t);
            end
        end
    end
    
    %+++++++++++++++  update Beta_t
    if beta0prior == 2 %normal prior for beta_0
        invA = x.Sigma_eta\eye(p); mu = invA*x.beta(:,2);
        L_eta = chol(invA + 1e-4*eye(p), 'lower');
        mu = L_eta\mu + randn([p,1]); x.beta(:,1) = L_eta'\mu;
        L_eta = chol(x.Sigma_eta, 'lower'); L_eta = L_eta\eye(p);
    else  %flat prior
        L_eta = chol(x.Sigma_eta, 'lower');
        x.beta(:,1) = L_eta*randn([p,1]) + x.beta(:,2);
        L_eta = L_eta\eye(p);
    end
    Pre = L_eta'*L_eta;
    for t = 1:T
        usesite = 1:N; if t==T; usesite = tind; end
        Sigma = X(indT==t&Tind==1,:)'*Dt{1,t};
        Deltat = squeeze(x.w(usesite,t,1)) - sum(X(indT==t&Tind==1,2:q).*squeeze(x.w(usesite,t,2:q)),2);
        Deltat0 = Deltat;
        if t > 1
            tmp = - squeeze(x.w(:,t-1,1)) + X(indT==t-1,:)*x.beta(:,t) + sum(X(indT==t-1,2:q).*squeeze(x.w(:,t-1,2:q)),2);
            Deltat = Deltat + tmp(usesite);
        end
        Mu = Sigma*Deltat + Pre*(x.beta(:,t));
        Sigma = Sigma*X(indT==t&Tind==1,:) + Pre;
        if t < T
            usesite2 = 1:N; if t == T-1; usesite2 = tind; end
            Xt = X(indT==t,:);
            Sigma1 = Xt(usesite2,:)'*Dt{1,t+1};
            Deltat1 = squeeze(x.w(usesite2,t+1,1)) - Deltat0(usesite2) - X(indT==t+1&Tind==1,:)*x.beta(:,t+2)...
                - sum(X(indT==t+1&Tind==1,2:q).*squeeze(x.w(usesite2,t+1,2:q)),2);
            Mu = Mu - Sigma1*Deltat1 + Pre*(x.beta(:,t+2));
            Sigma = Sigma + Sigma1*Xt(usesite2,:) + Pre;
        end
        Lo = chol(Sigma, 'lower');
        Mu = Lo\Mu; Mu = Mu + normrnd(0, 1, [length(Mu),1]);
        x.beta(:,t+1) = Lo'\Mu;
    end
    
    Xbeta = cell(1,T);
    for t = 1:T
        Xbeta{t} = X(indT==t,:)*x.beta(:,t+1);
    end
    
    %+++++++++++++++  update Sigma_eta
    A = zeros(p); for t = 2:(T+1); eta = x.beta(:,t) - x.beta(:,t-1); A = A + eta*eta'; end
    x.Sigma_eta = iwishrnd(IW_A+A, IW_nu+T);
    
    %+++++++++++++++  update w_tj
    for myupdateW = 1:updateW
        for t = 1:T
            for j = 2:q
                notj = 1:q; notj(j) = []; notj = notj(2:end);
                Sigma = repmat(X(indT==t,j), [1,N]).*Dt{1,t};
                Deltat = squeeze(x.w(:,t,1)) - sum(X(indT==t,notj).*squeeze(x.w(:,t,notj)),2) - Xbeta{t};
                Deltat0 = Deltat;
                w0 = zeros(N,1);
                if t > 1
                    Deltat = Deltat - x.w(:,t-1,1) + Xbeta{t-1} + sum(X(indT==t-1,2:q).*squeeze(x.w(:,t-1,2:q)),2);
                    w0 = x.w(:,t-1,j);
                end
                Mu = Sigma*Deltat + Dt{j,t}*w0;
                Sigma = repmat(X(indT==t,j)', [N,1]).*Sigma + Dt{j,t};
                if t < T
                    Sigma1 = repmat(X(indT==t,j), [1,N]).*Dt{1,t+1};
                    w0 = squeeze(x.w(:,t+1,1));
                    Deltat1 = w0 - Deltat0 - Xbeta{t+1} - sum(X(indT==t+1,2:q).*squeeze(x.w(:,t+1,2:q)),2);
                    Mu = Mu - Sigma1*Deltat1 + Dt{j,t+1}*squeeze(x.w(:,t+1,j));
                    Sigma = Sigma + repmat(X(indT==t,j)', [N,1]).*Sigma1 + Dt{j,t+1};
                end
                Lo = chol(Sigma, 'lower');
                Mu = Lo\Mu; Mu = Mu + randn([length(Mu),1]);
                x.w(:,t,j) = Lo'\Mu;
            end
        end
    end
    
    Delta = nan(N,T); L_w = cell(1,T);
    for t = 1:T
        Delta(:,t) = Xbeta{t} + sum(X(indT==t, 2:q).*squeeze(x.w(:,t,2:q)),2);
        L_w{t} = chol(Dt{1,t}, 'lower');
    end
    
    %+++++++++++++++ update w_tj for j=1 (log risk)
    for myupdateW1 = 1:updateW1
        for t = 1:T
            usesite = 1:N; if t == T; usesite = tind; end
            usesite2 = 1:N;
            if t == T-1
                usesite2 = tind;
            end
            for i = usesite %1:N
                vec = Delta(:,t); vec1 = Delta(:,t);
                if t > 1
                    vec = vec + x.w(:,t-1,1) - Delta(:,t-1);
                end
                
                if useMetropolis_eta ~= 1   % update Eta using Griddy Gibbs sampler
                    etamat = repmat(squeeze(x.w(:,t,1)), [1,len_eta]); etamat(i,:) = etavec;
                    tmp = 0;
                    if t < T
                        vec1 = vec1 + squeeze(x.w(:,t+1,1)) - Delta(:,t+1);
                        tmp =  - 0.5*sum((L_w{t+1}'*(etamat - repmat(vec1, [1,len_eta]))).^2, 1);
                    end
                    etamat = etamat - repmat(vec, [1,len_eta]);
                    loglike = loglike0_eta(:,i,t)' ... %Poisson part of likelihood
                        - 0.5*sum((L_w{t}'*(etamat - repmat(vec, [1,len_eta]))).^2, 1) + tmp;
                    MaxLogLike = max(loglike);
                    P = exp(loglike-MaxLogLike)/sum(exp(loglike-MaxLogLike));
                    % figure(1); plot(etavec,P)
                    U = rand(1);
                    cump = cumsum([0 P(1:(end-1))]);
                    i0 = sum(U > cump);
                    x.w(i,t,1) = etavec(1);
                    if i0 > 1
                        x.w(i,t,1) = etavec(i0-1) + gap_eta/P(i0)*(U-cump(i0));
                    end
                    
                else % update Eta using metropolis step, need tune proposal variance for acceptance rate
                    w1 = x.w(:,t,1); w1(i) = normrnd(x.w(i,t,1), exp(w_ls(i,t)));
                    loglik = - 0.5*sum((L_w{t}'*(x.w(usesite,t,1) - vec(usesite))).^2) + Y(i,t)*x.w(i,t,1) - E(i,t)*exp(x.w(i,t,1));
                    loglik1 = - 0.5*sum((L_w{t}'*(w1(usesite) - vec(usesite))).^2) + Y(i,t)*w1(i) - E(i,t)*exp(w1(i));
                    if t < T
                        tmp =  squeeze(x.w(usesite2,t+1,1)) - Delta(usesite2,t+1);
                        vec1(usesite2) = vec1(usesite2) + tmp;
                        loglik =  loglik - 0.5*sum((L_w{t+1}'*(x.w(usesite2,t,1) - vec1(usesite2))).^2);
                        loglik1 =  loglik1 - 0.5*sum((L_w{t+1}'*(w1(usesite2) - vec1(usesite2))).^2);
                    end
                    MH = exp(loglik1 - loglik);
                    u1 = rand(1);
                    if u1 <= MH
                        x.w(:,t,1) = w1;
                        w_accepts(i,t) = w_accepts(i,t)+1;
                    end
                end
                
            end
        end
    end
    
    %+++++++++++++++ update CAR parameters
    for myupdateParaCAR = 1:updateParaCAR 
        for t = 1:T
            usesite = 1:N;
            if t == T
                usesite = tind;
            end
            
            for j = 1:1 % update only for j == 1: random intercept
                vec = (j==1)*Delta(usesite,t);
                if t == 1
                    err0 = x.w(:,t,j) - vec;
                else
                    err0 = x.w(usesite,t,j) - x.w(usesite,t-1,j) -vec + (j==1)*Delta(usesite,t-1);
                end
                wDw = err0'*W(usesite,usesite)*err0;
                
                if nonspat ~= 1
                    loglike = lphi0 + (0.5/x.tau2(j,t))*wDw*phis;
                    MaxLogLike = max(loglike);
                    P = exp(loglike-MaxLogLike)/sum(exp(loglike-MaxLogLike));
                    % plot(phis,P)
                    U0 = rand(1);
                    cump = cumsum([0 P(1:(end-1))]);
                    i0 = sum(U0 > cump);
                    x.phi(j,t) = phis(1);
                    if i0 > 1
                        x.phi(j,t) = phis(i0-1) + gap_phi/P(i0)*(U0-cump(i0));
                    end
                end
                
                wDw = sum(diag(M(usesite,usesite)).*err0.^2) - x.phi(j,t)*wDw;
                bzeros = (invbetatau + 0.5*wDw)^-1;
                azeros = alphatau + 0.5*length(usesite);
                x.tau2(j,t) = 1./gamrnd(azeros, bzeros);
            end % =======================================================
        end
    end
    
    if(iter > burnin)
        matPara((iter-burnin),:) = [reshape(x.beta,[1,numel(x.beta)]), x.Sigma_eta(~~tril(x.Sigma_eta+5))',...
            reshape(x.tau2,[1,numel(x.tau2)]), reshape(x.phi,[1,numel(x.phi)]) ];
        % for predction at new locations at vind set for j=1, t=T
        Sigma = M-x.phi(1,T).*W; Lo = chol(Sigma(vind,vind), 'lower');
        Mu =  Sigma(vind, tind) * (  x.w(tind,T,1) - Delta(tind,T) - (x.w(tind,T-1,1) - Delta(tind,T-1))  );
        Mu = Lo\Mu;
        Mu = normrnd(0, sqrt(x.tau2(1,T)), [length(Mu),1]) - Mu;
        x.w(vind,T,1) = x.w(vind,T-1,1) - Delta(vind,T-1) + Delta(vind,T) + Lo'\Mu;
        Ws(iter-burnin,:) = reshape(x.w(vind,T,1), [1,numel(x.w(vind,T,1))]);
    end
    
    if ~mod(iter, batchLen)
        if reportrate == 1; disp(num2str(mean(w_accepts./batchLen), 2));  end
        % disp(num2str(mean(accepts./batchLen), 2))
        %++++++++++ scale tuning for adaptive MCMC
        if adaptMCMC == 1
            batchNum = batchNum+1;
            w_accepts = w_accepts./batchLen;
            w_rates(batchNum,:) = reshape(w_accepts, [1, numel(w_accepts)]);
            w_ls = w_ls + sign((w_accepts>objrate)-0.5).*min(0.01,1/sqrt(batchNum));
        end
        w_accepts = zeros(N,T);
    end
    
end


CPUtime = toc; CPUtime = t0 + CPUtime/60;
completed = 1;
fprintf('\n%d iterations are done with elapsed time %.2f minutes.\n', tot, CPUtime)
save(nam,'matPara','Ws','CPUtime', 'w_rates','truePara','trueW','Y','completed',...
    't0','iter0', 'batchNum','w_ls','w_accepts','x','E_BETApart','Zpart','Y_vind','vind','tind')
end
