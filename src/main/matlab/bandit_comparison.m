%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Bandit Comparison
%
% Estimate the probability of pulling the optimal lever and the regret
% as a function of the number of trials for different bandit algorithms.
% The rewards follow a Bernoulli distribution with unknown parameters.
%
% (c) 2014 Pedro A. Ortega <pedro.ortega@gmail.com>
%     2015 Modified by Andrew Forney <forns@cs.ucla.edu>
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Get ready.
close all;
clear all;
clc;

%% Script configuration.
%  Step 1: set any simulation configuration constants as you please.

T = 1000; % Number of timesteps.
N = 1000; % Number of Monte Carlo samples.
N_obs = 200; % Number of observational samples, evenly divided per arm.

%  Step 2: Select which algorithms to run by their name, and then choose
%          their respective labels in the plot
algorithms = {'thompsonRun', 'thompsonZRun', 'thompsonCausalRun'};
names  = {'TS', 'TS^{Z}', 'TS^{C}'};

colors = {[0.5 0.5 0], [0 0.5 0.5], [1 0 0], [0 1 0], [0 0 1], [0.3 0 0.7], [1 0.5 0], [1, 0.5, 1], [0.5, 1, 1], [1, 1, 0.5]};

%  Step 3: Choose problem parameters. Un-comment 1 theta below.
% theta := P(y | do(X), B, D)
    % D =    0    0    1    1
    % B =    0    1    0    1
theta   = [[0.1, 0.5, 0.4, 0.2]    % X = M1  -- Greedy Casino
           [0.5, 0.1, 0.2, 0.4]];  % X = M2
%theta   = [[0.5, 0.1, 0.2, 0.4]    % X = M1  -- Generous Casino
%           [0.1, 0.5, 0.4, 0.2]];  % X = M2
%theta   = [[0.4, 0.3, 0.3, 0.4]    % X = M1  -- Paradoxical Switching
%           [0.6, 0.1, 0.2, 0.6]];  % X = M2
%theta   = [[0.5, 0.5, 0.4, 0.2]    % X = M1  -- Sometimes Switch
%           [0.1, 0.1, 0.2, 0.15]];  % X = M2
%theta   = [[0.55, 0.55, 0.55, 0.55]    % X = M1  -- No Confounding
%           [0.45, 0.45, 0.45, 0.45]];  % X = M2
%theta   = [[0, 1, 1, 0]    % X = M1  -- Inevitable Regret
%           [1, 0, 0, 1]];  % X = M2

%  Step 4: Run this script! Your configurations are complete and we'll take
%          it from here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize.
% pObs := P(Y | X)
    % Y =    0     1
pObs    = [[(1-theta(1, 1) + 1-theta(1, 4)) * N_obs / 4, (theta(1, 1) + theta(1, 4)) * N_obs / 4]    % X = M1
           [(1-theta(2, 2) + 1-theta(2, 3)) * N_obs / 4, (theta(2, 2) + theta(2, 3)) * N_obs / 4]];  % X = M2

% Total number of algorithms.
nAlgs = length(algorithms);

% Sums of actions, rewards, and times best action chosen.
ActionSum = zeros(nAlgs, T);
RewardSum = zeros(nAlgs, T);
ProbSum   = zeros(nAlgs, T);
CondSum   = zeros(nAlgs, 4); % B, D binary

RoundRewards = zeros(nAlgs, N);
RoundRegrets = zeros(nAlgs, N);

% Open figure.
h = figure();
set(h, 'Position', [100, 100, 1400, 500]);
    
for alg=1:nAlgs
    % Get handle of algorithm to run.
    fprintf('Running algorithm: %s\n', algorithms{alg});
    fhandle = str2func(algorithms{alg});
    currentFactors = zeros(3, T);

    %% Generate Monte Carlo simulations.
    for n=1:N
        % Determine covariates for this run
        for t=1:T
            B = rand <= 0.5;
            D = rand <= 0.5;
            Z = xor(B, D) + 1;
            currentFactors(:, t) = [B, D, Z];
        end
        
        % Generate one run.
        [Action, Reward, Prob, Conds] = fhandle(theta, T, currentFactors, pObs);

        % Collect stats.
        RewardSum(alg, :) = RewardSum(alg, :) + Reward;
        ActionSum(alg, :) = ActionSum(alg, :) + Action;
        ProbSum(alg, :)   = ProbSum(alg, :) + Prob;
        CondSum(alg, :)   = CondSum(alg, :) + Conds;
        
        RoundRewards(alg, n) = sum(Reward);
        RoundRegrets(alg, n) = (max(theta(:,1))*Conds(1) + max(theta(:,2))*Conds(2) + max(theta(:,3))*Conds(3) + max(theta(:,4))*Conds(4)) - sum(Reward);
        
        % Report progress.
        if (mod(n,100) == 0)
            fprintf('Samples: %5d\n', n);
        end
    end
    
    %% Monte Carlo estimates.
    PAction = ProbSum(alg, :)/N;
    Regret  = (max(theta(:,1))*(CondSum(alg,1)/(N)) + max(theta(:,2))*(CondSum(alg,2)/(N)) + max(theta(:,3))*(CondSum(alg,3)/(N)) + max(theta(:,4))*(CondSum(alg,4)/(N)))*[1:T]/T - cumsum(RewardSum(alg, :)/N);
    

    %% Plot.
    
    % axes for plots
    probAxis   = [0, T, 0.4, 1.0];
    regretAxis = [0, T, 0, 300];
    N_PLOTS = 2;
    
    % Plot probability of pulling the best action.
    subplot(1,N_PLOTS,1);
    plot(PAction, 'Color', colors{alg});
    hold on;
    title('Probability of Optimal Action');
    xlabel('Trial');
    ylabel('Probability');
    axis square;
    axis(probAxis);
    
    % Plot regret.
    subplot(1,N_PLOTS,2);
    plot(Regret, 'Color', colors{alg});
    hold on;
    title('Regret');
    xlabel('Trial');
    ylabel('Cum. Regret');
    axis square;
    axis(regretAxis);
    
    drawnow;
end

%% Finish.
% Place legend.
subplot(1,N_PLOTS,1);
legend(names, 'Location', 'SouthEast');
subplot(1,N_PLOTS,2);
legend(names, 'Location', 'SouthEast');

% END OF SCRIPT