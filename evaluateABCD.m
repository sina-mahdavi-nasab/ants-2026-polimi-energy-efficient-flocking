function f = evaluateABCD(ABCD_mat)
    addpath(genpath(pwd))
    %rules = unflattenABCD(ABCD_vec);
   
    % Average efficiency over nSwarms independent runs
    %eff = zeros(1,nSwarms);
    %for k = 1:nSwarms
        %eff = simulation(ABCD_vec, nSteps, rules);
        %f = -eff;
    %end
    %f = -mean(eff);                 % CMA-ES MINIMISES → negate
    %if isnan(f) || ~isfinite(f), f = 1e99; end

    %  ABCD_mat : 2448 × λ   (λ = 30)
%  F        : 1 × λ      (negative efficiencies)
% ------------------------------------------------------------------

        % Set the maximum number of parallel workers
    nWorkers = 6;
      % Start parallel pool with 4 workers

    lambda = size(ABCD_mat, 2);
    f = NaN(1, lambda);  % pre-allocate
    % nWorkers = max(parcluster('local').NumWorkers, lambda);
    

    % --- fitness for every individual, one swarm each --------------
    parfor (k = 1:lambda, nWorkers)       % requires Parallel TB

        rules = unflattenABCD(ABCD_mat(:,k));
        seed1 = randi(2^31-1);
        seed2 = randi(2^31-1);
        seed3 = randi(2^31-1);
        eff1   = simulation_free_global_mod(rules,seed1);
        eff2   = simulation_free_global_mod(rules,seed2);
        eff3   = simulation_free_global_mod(rules,seed3);
        eff   = sort([eff1, eff2, eff3]);
        f(k)  = -eff(2);                % CMA-ES minimises
    end

    
end
function R = unflattenABCD(v)
    idx=0; take=@(n) reshape(v(idx+(1:n)),[],1);
    grab=@(r,c) reshape( take(r*c), r,c );
    R.A1 = grab(14,14); idx=idx+196;
    R.A2 = grab(14,14); idx=idx+196;
    R.A3 = grab(14, 2); idx=idx+28 ;
    R.B1 = grab(14,14); idx=idx+196;
    R.B2 = grab(14,14); idx=idx+196;
    R.B3 = grab(14, 2); idx=idx+28 ;
    R.C1 = grab(14,14); idx=idx+196;
    R.C2 = grab(14,14); idx=idx+196;
    R.C3 = grab(14, 2); idx=idx+28 ;
    R.D1 = grab(14,14); idx=idx+196;
    R.D2 = grab(14,14); idx=idx+196;
    R.D3 = grab(14, 2);
end