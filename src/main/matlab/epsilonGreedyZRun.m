%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Epsilon-Greedy-Z Bandit Player.
% Contextual epsilon-greedy player, treating intuition as context.
%
% (c) 2014 Pedro A. Ortega <pedro.ortega@gmail.com>
%     2015 Modified by Andrew Forney
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Action, Reward, Prob, Conds] = epsilonGreedyZRun(theta, T, allFactors, pObs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize.
s = [[1, 1]
     [1, 1]];
f = [[1, 1]
     [1, 1]];

Action = zeros(1, T);
Reward = zeros(1, T);
Prob   = zeros(1, T);
Conds  = zeros(1, 4); % B, D binary

epsilon = 0.05;

%% Execute one run.
for t=1:T
    roundFactors = allFactors(:,t);
    B = roundFactors(1);
    D = roundFactors(2);
    Z = roundFactors(3);
    zPrime = 3 - Z; % If z = 1, zPrime = 2; if z = 2, zPrime = 1
    covariateIndex = B + D * 2 + 1;
    Conds(covariateIndex) = Conds(covariateIndex) + 1;
    
    exploring = (rand < epsilon);
    
    p_Y_doX_Z = [[s(1, 1) / (s(1, 1) + f(1, 1)), s(1, 2) / (s(1, 2) + f(1, 2))]   % Z = M1
                 [s(2, 1) / (s(2, 1) + f(2, 1)), s(2, 2) / (s(2, 2) + f(2, 2))]]; % Z = M2
    
    % Choose action.
    if exploring
        action = (rand < 0.5) + 1;
    else
        [highVal, action] = max([s(Z, 1) / (s(Z, 1) + f(Z, 1)), s(Z, 2) / (s(Z, 2) + f(Z, 2))]);
    end
    currentTheta = theta(action, covariateIndex);

    % Pull lever. We'll find the index of theta through some clever maths
    % with the value of B and D (see covariate index)
    reward = rand <= currentTheta;
    
    % Update.
    s(Z, action) = s(Z, action) + reward;
    f(Z, action) = f(Z, action) + 1 - reward;
    
    % Record.
    Action(t) = (action == 1);
    Reward(t) = reward;
    [bestVal, bestAction] = max([theta(1, covariateIndex), theta(2, covariateIndex)]);
    Prob(t) = action == bestAction;
end

