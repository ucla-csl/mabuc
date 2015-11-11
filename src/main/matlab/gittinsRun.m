%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Gittins Bandit Player.
%
% (c) 2014 Pedro A. Ortega <pedro.ortega@gmail.com>
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Action, Reward, Prob, Conds] = gittinsRun(theta, T, allFactors, pObs)

%% Configuration.
gittinsFile = 'gittins.txt';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load indices.
persistent gittins
if isempty(gittins)
    gittins = dlmread(gittinsFile);
end

%% Initialize.
s = [0, 0];
f = [0, 0];

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
    covariateIndex = B + D * 2 + 1;
    Conds(covariateIndex) = Conds(covariateIndex) + 1;
    
    % Choose action.
    gIdx = [gittins(f(1)+1, s(1)+1), gittins(f(2)+1, s(2)+1)];
    [maxVal, action] = max(gIdx);

    % Pull lever.
    reward = rand <= theta(action, covariateIndex);

    % Update.
    s(action) = s(action) + reward;
    f(action) = f(action) + 1 - reward;

    % Record.
    Action(t) = (action == 1);
    Reward(t) = reward;
    [bestVal, bestAction] = max([theta(1, covariateIndex), theta(2, covariateIndex)]);
    Prob(t) = action == bestAction;
end

