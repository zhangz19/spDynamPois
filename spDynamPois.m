function [out] = spDynamPois(Y, X, E, W, ev)

[N, p, T] = size(X); % number of fixed effects
q = 1; % number of random effects: q=1 means varying-intercept only.  
% fprintf('N = %d, T = %d, p = %d, q = %d\n', [N, T, p, q])

%======================================= set hyperparameters for priors
meantau2 = 0.01; vartau2 = 10^4;  % specify the prior mean and variance for tau2
alphatau = 2+meantau2^2/vartau2;
invbetatau = meantau2*(alphatau-1);
IW_nu = p+2;  IW_A = eye(p);
% alphatau = 0; invbetatau = 0; IW_nu = p+1 + 1e-3; IW_A = 1e-3*eye(p); %p+3+ 1e-10*

if ev.adaptMCMC;  objrate = 0.44; batchLen = 50; batchNum = 0; batchTot = ev.niter/batchLen;  end

if ~ ev.useMetropolis_eta  %then use Griddy sampler, some terms in likelihood can be pre-calculated.
    gap_eta = 1e-2; etavec = (-4):gap_eta:4; % may need to change the upper bound  5
    len_eta = length(etavec);
    loglike0_eta = zeros(len_eta, N, T);
    for i = 1:N; for t = 1:T;  loglike0_eta(:,i,t) = Y(i,t).*etavec - E(i,t).*exp(etavec);  end; end
end

M = diag(max(1, sum(W, 1)));
invM = inv(M); eigs = eig(sqrt(invM)*W*sqrt(invM));
lphi = max(1/min(eigs),-1); uphi = 1/max(eigs);
gap_phi = 1e-2; phis = (lphi+gap_phi):gap_phi:(uphi-gap_phi); len = numel(phis);
lphi0 = zeros(1,len);
for i = 1:len;  lphi0(i) = 0.5*sum(log(eig(M-phis(i)*W)));  end

% initialize: start with non-spatial model
Yobs = log((Y + 0.5*(Y==0))./E);   pa.beta = zeros(p,T+1);   pa.tau2 = 0.04*ones(q,T);
pa.w = zeros(N,T,q);
pa.phi = 0.9 + zeros(q,T); %x.phi(1,1) = 0;
pa.Sigma_eta = 0.01*eye(p);
for t = 1:T;   pa.beta(:,t+1) = (X(:,:,t)'*X(:,:,t))\(X(:,:,t)'*Yobs(:,t));   end
% now recode x.w(:,1,:) to be the log risk for updating which facilitates the computation
pa.w(:,:,1) = Yobs;  %initial

% ============================ initial log-std for proposal ==================================
facw = .01;  % Need to tuning the initial scale for reasonable acceptance at the beginning.
w_ls = log(facw*(abs(pa.w(:,:,1)) + 1e-6)); %repmat(log(sqrt(.02*diag(cov(x.w')))), [1,T]); %
% ======================================================================================

Zpart = ones(N,T);
for t = 1:T
    [b, ~, stats] = glmfit(X(:,2:end,t), Y(:,t), 'poisson','offset', log(E(:,t)));
    pa.beta(:,t+1) = b; %(X(:,:,t)'*X(:,:,t))\(X(:,:,t)'*Yobs(:,t));
    Zpart0 = exp(sum(X(:,1:q,t).*squeeze(pa.w(:,t,:)),2));
    Zpart(:,t) = Zpart0;
end
pa.beta = pa.beta + normrnd(0,1,size(pa.beta)).*abs(pa.beta)*0.001;

E_BETApart = zeros(N,T); %stores E*exp(X*beta)
for t = 1:T;  Delta = X(:,:,t)*pa.beta(:,t+1); E_BETApart(:,t) = E(:,t).*exp(Delta);   end

if ev.nonspat == 1;   pa.phi = pa.phi*0; M = eye(size(M,1));   end % nonspatial case

tind = 1:N;  vind = [];   %tind: index for training data;  vind: index for validation data
if ev.crossValidation  % if true, random subset of locations for the last time point will be hold out for predicting 
    nv = ceil(N*ev.crossValidationPercent);    
    vind = randperm(N);  vind = sort(vind(1:nv));  tind(vind) = [];
end

% MCMC running
npara = numel(pa.beta) + numel(pa.phi) + numel(pa.tau2) + numel(pa.Sigma_eta(~~tril(pa.Sigma_eta+5)));
out.matPara = nan((ev.niter-ev.burnin), npara);
out.Ws = nan((ev.niter-ev.burnin), numel(pa.w(vind,T,1)));
w_accepts = zeros(N,T); 
w_rates = zeros(batchTot, N*T);
% beta_ls = log(0.3*abs(x.beta(:,2:end) + 10*exp(-5)));

tic
for iter = 1:ev.niter
    if ev.verbose==1; fprintf('%6d', iter); if(~mod(iter,20)); fprintf('\n'); end; end
    
    %+++++++++++++++  construct CAR covariance matrices
    Dt = cell(q,T);
    for t = 1:T
        for j = 1:q
            if t ~= T; Dt{j,t} = (M - pa.phi(j,t)*W)/pa.tau2(j,t);
            else Dt{j,t} = (M(tind, tind) - pa.phi(j,t)*W(tind, tind))/pa.tau2(j,t);  %for t=T
            end
        end
    end
    
    %+++++++++++++++  update Beta_t
    if ev.beta0prior == 2 %normal prior for beta_0
        invA = pa.Sigma_eta\eye(p); mu = invA*pa.beta(:,2);
        L_eta = chol(invA + 1e-4*eye(p), 'lower');
        mu = L_eta\mu + randn([p,1]); pa.beta(:,1) = L_eta'\mu;
        L_eta = chol(pa.Sigma_eta, 'lower'); L_eta = L_eta\eye(p);
    else  %flat prior
        L_eta = chol(pa.Sigma_eta, 'lower');
        pa.beta(:,1) = L_eta*randn([p,1]) + pa.beta(:,2);
        L_eta = L_eta\eye(p);
    end
    Pre = L_eta'*L_eta;
    for t = 1:T
        usesite = 1:N;  if t==T; usesite = tind; end
        Sigma = X(usesite,:,t)'*Dt{1,t};
        Deltat = squeeze(pa.w(usesite,t,1)) - sum(X(usesite,2:q,t).*squeeze(pa.w(usesite,t,2:q)),2);
        Deltat0 = Deltat;
        if t > 1
            tmp = - squeeze(pa.w(:,t-1,1)) + X(:,:,t-1)*pa.beta(:,t) + sum(X(:,2:q,t-1).*squeeze(pa.w(:,t-1,2:q)),2);
            Deltat = Deltat + tmp(usesite);
        end
        Mu = Sigma*Deltat + Pre*(pa.beta(:,t));
        Sigma = Sigma*X(:,:,t) + Pre;
        if t < T
            usesite2 = 1:N; if t == T-1; usesite2 = tind; end
            Sigma1 = X(usesite2,:,t)'*Dt{1,t+1};
            Deltat1 = squeeze(pa.w(usesite2,t+1,1)) - Deltat0(usesite2) - X(usesite2,:,t+1)*pa.beta(:,t+2)...
                - sum(X(usesite2,2:q,t+1).*squeeze(pa.w(usesite2,t+1,2:q)),2);
            Mu = Mu - Sigma1*Deltat1 + Pre*(pa.beta(:,t+2));
            Sigma = Sigma + Sigma1*X(usesite2,:,t) + Pre;
        end
        Lo = chol(Sigma, 'lower');
        Mu = Lo\Mu; Mu = Mu + normrnd(0, 1, [length(Mu),1]);
        pa.beta(:,t+1) = Lo'\Mu;
    end
    
    Xbeta = cell(1,T);  for t = 1:T;  Xbeta{t} = X(:,:,t)*pa.beta(:,t+1);  end
    
    %+++++++++++++++  update Sigma_eta
    A = zeros(p); for t = 2:(T+1); eta = pa.beta(:,t) - pa.beta(:,t-1); A = A + eta*eta'; end
    pa.Sigma_eta = iwishrnd(IW_A+A, IW_nu+T);
    
    %+++++++++++++++  update w_tj
    for t = 1:T
        for j = 2:q
            notj = 1:q; notj(j) = []; notj = notj(2:end);
            Sigma = repmat(X(:,j,t), [1,N]).*Dt{1,t};
            Deltat = squeeze(pa.w(:,t,1)) - sum(X(:,notj,t).*squeeze(pa.w(:,t,notj)),2) - Xbeta{t};
            Deltat0 = Deltat;
            w0 = zeros(N,1);
            if t > 1
                Deltat = Deltat - pa.w(:,t-1,1) + Xbeta{t-1} + sum(X(:,2:q,t-1).*squeeze(pa.w(:,t-1,2:q)),2);
                w0 = pa.w(:,t-1,j);
            end
            Mu = Sigma*Deltat + Dt{j,t}*w0;
            Sigma = repmat(X(:,j,t)', [N,1]).*Sigma + Dt{j,t};
            if t < T
                Sigma1 = repmat(X(:,j,t), [1,N]).*Dt{1,t+1};
                w0 = squeeze(pa.w(:,t+1,1));
                Deltat1 = w0 - Deltat0 - Xbeta{t+1} - sum(X(:,2:q,t+1).*squeeze(pa.w(:,t+1,2:q)),2);
                Mu = Mu - Sigma1*Deltat1 + Dt{j,t+1}*squeeze(pa.w(:,t+1,j));
                Sigma = Sigma + repmat(X(:,j,t)', [N,1]).*Sigma1 + Dt{j,t+1};
            end
            Lo = chol(Sigma, 'lower');
            Mu = Lo\Mu; Mu = Mu + randn([length(Mu),1]);
            pa.w(:,t,j) = Lo'\Mu;
        end
    end
    
    Delta = nan(N,T); L_w = cell(1,T);
    for t = 1:T
        Delta(:,t) = Xbeta{t} + sum(X(:, 2:q,t).*squeeze(pa.w(:,t,2:q)),2);
        L_w{t} = chol(Dt{1,t}, 'lower');
    end
    
    %+++++++++++++++ update w_tj for j=1 (log risk)
    for t = 1:T
        usesite = 1:N; if t == T; usesite = tind; end
        usesite2 = 1:N;
        if t == T-1;  usesite2 = tind;  end
        for i = usesite %1:N
            vec = Delta(:,t); vec1 = Delta(:,t);
            if t > 1;   vec = vec + pa.w(:,t-1,1) - Delta(:,t-1);   end
            
            if ~ev.useMetropolis_eta  % update Eta using Griddy Gibbs sampler
                etamat = repmat(squeeze(pa.w(:,t,1)), [1,len_eta]); etamat(i,:) = etavec;
                tmp = 0;
                if t < T
                    vec1 = vec1 + squeeze(pa.w(:,t+1,1)) - Delta(:,t+1);
                    tmp =  - 0.5*sum((L_w{t+1}'*(etamat - repmat(vec1, [1,len_eta]))).^2, 1);
                end
                etamat = etamat - repmat(vec, [1,len_eta]);
                loglike = loglike0_eta(:,i,t)' ... %Poisson part of likelihood
                    - 0.5*sum((L_w{t}'*(etamat - repmat(vec, [1,len_eta]))).^2, 1) + tmp;
                MaxLogLike = max(loglike);
                P = exp(loglike-MaxLogLike)/sum(exp(loglike-MaxLogLike));
                U = rand(1);
                cump = cumsum([0 P(1:(end-1))]);
                i0 = sum(U > cump);
                pa.w(i,t,1) = etavec(1);
                if i0 > 1;   pa.w(i,t,1) = etavec(i0-1) + gap_eta/P(i0)*(U-cump(i0));   end
                
            else % update Eta using metropolis step, need tune proposal variance for acceptance rate
                w1 = pa.w(:,t,1); w1(i) = normrnd(pa.w(i,t,1), exp(w_ls(i,t)));
                loglik = - 0.5*sum((L_w{t}'*(pa.w(usesite,t,1) - vec(usesite))).^2) + Y(i,t)*pa.w(i,t,1) - E(i,t)*exp(pa.w(i,t,1));
                loglik1 = - 0.5*sum((L_w{t}'*(w1(usesite) - vec(usesite))).^2) + Y(i,t)*w1(i) - E(i,t)*exp(w1(i));
                if t < T
                    tmp =  squeeze(pa.w(usesite2,t+1,1)) - Delta(usesite2,t+1);
                    vec1(usesite2) = vec1(usesite2) + tmp;
                    loglik =  loglik - 0.5*sum((L_w{t+1}'*(pa.w(usesite2,t,1) - vec1(usesite2))).^2);
                    loglik1 =  loglik1 - 0.5*sum((L_w{t+1}'*(w1(usesite2) - vec1(usesite2))).^2);
                end
                MH = exp(loglik1 - loglik);
                u1 = rand(1);
                if u1 <= MH;   pa.w(:,t,1) = w1;  w_accepts(i,t) = w_accepts(i,t)+1;   end
            end
            
        end
    end
    
    %+++++++++++++++ update CAR parameters
    for t = 1:T
        usesite = 1:N;
        if t == T;   usesite = tind;   end
        
        for j = 1:1 % update only for j == 1: random intercept
            vec = (j==1)*Delta(usesite,t);
            if t == 1
                err0 = pa.w(:,t,j) - vec;
            else
                err0 = pa.w(usesite,t,j) - pa.w(usesite,t-1,j) -vec + (j==1)*Delta(usesite,t-1);
            end
            wDw = err0'*W(usesite,usesite)*err0;
            
            if ev.nonspat ~= 1
                loglike = lphi0 + (0.5/pa.tau2(j,t))*wDw*phis;
                MaxLogLike = max(loglike);
                P = exp(loglike-MaxLogLike)/sum(exp(loglike-MaxLogLike));
                % plot(phis,P)
                U0 = rand(1);
                cump = cumsum([0 P(1:(end-1))]);
                i0 = sum(U0 > cump);
                pa.phi(j,t) = phis(1);
                if i0 > 1;   pa.phi(j,t) = phis(i0-1) + gap_phi/P(i0)*(U0-cump(i0));   end
            end
            
            wDw = sum(diag(M(usesite,usesite)).*err0.^2) - pa.phi(j,t)*wDw;
            bzeros = (invbetatau + 0.5*wDw)^-1;
            azeros = alphatau + 0.5*length(usesite);
            pa.tau2(j,t) = 1./gamrnd(azeros, bzeros);
        end
    end
    
    if(iter > ev.burnin)
        out.matPara((iter-ev.burnin),:) = [reshape(pa.beta,[1,numel(pa.beta)]), pa.Sigma_eta(~~tril(pa.Sigma_eta+5))',...
            reshape(pa.tau2,[1,numel(pa.tau2)]), reshape(pa.phi,[1,numel(pa.phi)]) ];
        % for predction at new locations at vind set for j=1, t=T
        Sigma = M-pa.phi(1,T).*W; Lo = chol(Sigma(vind,vind), 'lower');
        Mu =  Sigma(vind, tind) * (  pa.w(tind,T,1) - Delta(tind,T) - (pa.w(tind,T-1,1) - Delta(tind,T-1))  );
        Mu = Lo\Mu;
        Mu = normrnd(0, sqrt(pa.tau2(1,T)), [length(Mu),1]) - Mu;
        pa.w(vind,T,1) = pa.w(vind,T-1,1) - Delta(vind,T-1) + Delta(vind,T) + Lo'\Mu;
        out.Ws(iter-ev.burnin,:) = reshape(pa.w(vind,T,1), [1,numel(pa.w(vind,T,1))]);
    end
    
    if ~mod(iter, batchLen)
        if ev.verbose == 1; disp(num2str(mean(w_accepts./batchLen), 2));  end
        % disp(num2str(mean(accepts./batchLen), 2))
        %++++++++++ scale tuning for adaptive MCMC
        if ev.adaptMCMC == 1
            batchNum = batchNum+1;
            w_accepts = w_accepts./batchLen;
            w_rates(batchNum,:) = reshape(w_accepts, [1, numel(w_accepts)]);
            w_ls = w_ls + sign((w_accepts>objrate)-0.5).*min(0.01, 1/sqrt(batchNum));
        end
        w_accepts = zeros(N,T);
    end
    
end


out.CPUtime = toc/60;  % in minutes
fprintf('\n%d iterations are done with elapsed time %.2f minutes.\n', ev.niter, out.CPUtime)

end
