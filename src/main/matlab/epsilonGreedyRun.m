%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Epsilon-Greedy Bandit Player.
% Naive player with no apprehension of each round's intuition.
%
% (c) 2014 Pedro A. Ortega <pedro.ortega@gmail.com>
%     2015 Modified by Andrew Forney
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Action, Reward, Prob, Conds] = epsilonGreedyRun(theta, T, allFactors, pObs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize.
s = [0, 0];
f = [0, 0];

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
    covariateIndex = B + D * 2 + 1;
    Conds(covariateIndex) = Conds(covariateIndex) + 1;
    
    exploring = (rand < epsilon);
    
    % Choose action.
    if exploring
        action = (rand < 0.5) + 1;
    else
        [highVal, action] = max([s(1) / (s(1) + f(1)), s(2) / (s(2) + f(2))]);
    end
    currentTheta = theta(action, covariateIndex);

    % Pull lever. We'll find the index of theta through some clever maths
    % with the value of B and D (see covariate index)
    reward = rand <= currentTheta;

    % Update.
    s(action) = s(action) + reward;
    f(action) = f(action) + 1 - reward;

    % Record.
    Action(t) = (action == 1);
    Reward(t) = reward;
    [bestVal, bestAction] = max([theta(1, covariateIndex), theta(2, covariateIndex)]);
    Prob(t) = action == bestAction;
end

