%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Thompson Sampling Causal Bandit Player.
% Causally empowered TS bandit player with (X's indicate presence):
% [X] Weighting applied based on ETT inequality?
% [X] ETT Seeding based on input observations?
% [X] Values for seeded ETT quantities are locked from seeding? 
%
%     2015 by Andrew Forney
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Action, Reward, Prob, Conds] = thompsonCausalRun(theta, T, allFactors, pObs)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize.
s = [[1, 1]
     [1, 1]];
f = [[1, 1]
     [1, 1]];
p_X = [sum(pObs(1, :)) / sum(pObs(:)), sum(pObs(2, :)) / sum(pObs(:))];
p_Y_X = [pObs(1, 2) / sum(pObs(1, :)), pObs(2, 2) / sum(pObs(2, :))];
zCount = [0, 0];

% Seed P(y | do(X), z) with observations, whenever X = Z
s(1, 1) = pObs(1, 2);
s(2, 2) = pObs(2, 2);
f(1, 1) = pObs(1, 1);
f(2, 2) = pObs(2, 1);

Action = zeros(1, T);
Reward = zeros(1, T);
Prob   = zeros(1, T);
Conds  = zeros(1, 4); % B, D binary

%% Execute one run.
for t=1:T
    roundFactors = allFactors(:,t);
    B = roundFactors(1);
    D = roundFactors(2);
    Z = roundFactors(3);
    zPrime = 3 - Z; % If z = 1, zPrime = 2; if z = 2, zPrime = 1
    covariateIndex = B + D * 2 + 1;
    Conds(covariateIndex) = Conds(covariateIndex) + 1;
    
    % Compute necessary stats.
    zCount(Z) = zCount(Z) + 1;
    p_Z = [zCount(1) / sum(zCount), zCount(2) / sum(zCount)];
    p_Y_doX_Z = [[s(1, 1) / (s(1, 1) + f(1, 1)), s(1, 2) / (s(1, 2) + f(1, 2))]
                 [s(2, 1) / (s(2, 1) + f(2, 1)), s(2, 2) / (s(2, 2) + f(2, 2))]];

    % Q1 = E(y_x' | x)  [Counter-intuition]
    Q1 = p_Y_doX_Z(Z, zPrime);
    % Q2 = E(y_x | x) [Intuition]
    Q2 = p_Y_X(Z);

    % Perform weighting
    bias = abs(Q1 - Q2);
    w = [1, 1];
    if isnan(bias)
        weighting = 1;
    else
        weighting = 1 - bias;
    end
    if Q1 > Q2
        w(Z) = weighting;
    else
        w(zPrime) = weighting;
    end
    
    % Choose action.
    theta_hat = [betarnd(s(Z, 1), f(Z, 1))*w(1), betarnd(s(Z, 2),f(Z, 2))*w(2)];
    [maxVal, action] = max(theta_hat);
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
