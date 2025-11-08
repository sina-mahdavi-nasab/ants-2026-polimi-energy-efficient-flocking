clc;
clear all;
close all;
addpath(genpath(pwd))

 %% ------------  Global Parameters  --------------------------------
    %N_ROBOTS        = 20;          % robots per swarm
    %N_SWARMS        = 1;          % fitness samples per individual
    GEN_MAX         = 100;         % CMA‑ES generations
    STEPS_PER_TRIAL = 400;         % sim timesteps per swarm

%% 1)  CMA‑ES hyper‑parameters ---------------------------------------
    Nvars = 1680;
    % ABCD_init = 10*rand(Nvars,1)-5;
    struct_ABCD_init = load('bestABCD.mat');
    ABCD_init = struct_ABCD_init.bestABCD;
    sigma0 = 1;
    
    opts = cmaes('defaults'); % We use the attributes of cames now
    opts.PopSize = 30;
    opts.MaxIter = GEN_MAX;
    opts.Restarts = 3;
    opts.DispModulo = 1; %Displays information
    opts.LBounds = -5 ;
    opts.UBounds = 5 ;
    opts.EvalParallel = 'yes';
    opts.SaveVariables = 'on'; %change to 'on' or 'final' if the simulation takes longer and we want to save files to resume later
    opts.SaveFilename  = 'runABCD.mat';
    opts.Resume        = 'on';
    %% 2)  Launch evolutionary search ------------------------------------
    fprintf('\n=====  Optimising ABCD Hebbian coefficients with CMA‑ES  =====\n');
    tic
    fitnesshandle ='evaluateABCD'; % CMAES take string as first argument
    [bestABCD,bestF,~,stopflag,out] = cmaes(fitnesshandle, ABCD_init, sigma0, opts);
     
    fprintf('\nFinished in %.1f s, best cost %.4g\n', toc, bestF);
    save('bestABCD.mat','bestABCD','bestF');


