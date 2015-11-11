%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Exp3 Bandit Player.
% Naive Exp3 bandit player, ignoring intuition at each round.
%
% (c) Jeremy Kun
% http://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorith
% m/
% Modified by Andrew Forney
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Action, Reward, Prob, Conds] = exp3Run(theta, T, allFactors, pObs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize.
weights = ones(1, 2); % 2 actions
gamma   = 0;

Action = zeros(1, T);
Reward = zeros(1, T);
Prob   = zeros(1, T);
Conds  = zeros(1, 4); % B, D binary

%% Execute one run.
for t=1:T
    % Convenient notation for covariates
    roundFactors = allFactors(:,t);
    B = roundFactors(1);
    D = roundFactors(2);
    Z = roundFactors(3);
    covariateIndex = B + D * 2 + 1;
    Conds(covariateIndex) = Conds(covariateIndex) + 1;
    
    % Setup distribution
    currentSum = sum(weights);
    distribution = weights(:)';
    for w=1:length(weights)
        currentWeight = weights(w);
        distribution(w) = (1.0 - gamma) * (currentWeight / currentSum) + (gamma / length(weights));
    end
    
    % Choose action.
    action = 1;
    choice = rand;
    for w=1:length(weights)
        choice = choice - weights(w);
        if choice <= 0
            break;
        end
        action = action + 1;
    end
    
    currentTheta = theta(action, covariateIndex);

    % Pull lever. We'll find the index of theta through some clever maths
    % with the value of B and D (see covariate index)
    reward = rand <= currentTheta;

    % Update.
    estimatedReward = 1.0 * reward / distribution(action);
    weights(action) = weights(action) * exp(estimatedReward * gamma / length(weights));

    % Record.
    Action(t) = (action == 1);
    Reward(t) = reward;
    [bestVal, bestAction] = max([theta(1, covariateIndex), theta(2, covariateIndex)]);
    Prob(t) = action == bestAction;
end

