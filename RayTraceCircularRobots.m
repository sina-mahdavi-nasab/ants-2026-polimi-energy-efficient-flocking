function [yVals, xVals, Psm] = RayTraceCircularRobots( ...
                    robots, radius, Uinf, ...
                    xRange, yRange, Nx, Ny, useGPU)

recoveryRate =  1;     % fraction of gap recovered per step outside circles
percentDrop = 0.25;      % wind intensity drop on hitting robot
maxWallSpan = 0.7;    %lower -> more wall effect
minPowerX = 30;
alpha = 0.5;  % kernel decay rate 1(larger = faster decay)
beta = 0.5;
xSmoothing1 = 100;  %lower -> more smoothing
ySmoothing1 = 50;
xSmoothing2 = 50;
ySmoothing2 = 50;
minPowerY = 10;

% RAYTRACECIRCULARROBOTSFAST
% Vectorised & GPU-ready version of your wind-ray routine.
%
% • Works on CPU or GPU   (set useGPU = true and own a PCT-supported card)
% • Keeps EXACT same physics: first-entry drop, in-circle hold,
%   re-drop when switching circles, exponential recovery outside.
%
% -------------------------------------------------------------------------

% -- GRID -----------------------------------------------------------------
xVals = linspace(xRange(1), xRange(2), Nx);
yVals = linspace(yRange(1), yRange(2), Ny);
ySpan = yRange(2)-yRange(1);
xSpan = xRange(2)-xRange(1);

if useGPU
    xVals = gpuArray(single(xVals));
    yVals = gpuArray(single(yVals));
end
dx   = xVals(2) - xVals(1);
dy   = yVals(2) - yVals(1);

% -- BROADCAST DISTANCES (Ny × Nx × M) -----------------------------------
[X, Y] = meshgrid(xVals, yVals);                     % Ny×Nx
xC = reshape(robots(:,1), 1, 1, []);                 % 1×1×M
yC = reshape(robots(:,2), 1, 1, []);                 % 1×1×M
distPages = hypot(X - xC, Y - yC);                   % Ny×Nx×M

% • idxMat  : index of CLOSEST robot at each grid cell (0 = none)
% • inMask  : logical mask where any robot covers the cell
[distMin, idxMat] = min(distPages, [], 3);           % Ny×Nx
inMask            = distMin < radius;                % Ny×Nx logical
idxMat(~inMask)   = 0;                               % outside ⇒ 0

% -- POWER FIELD ----------------------------------------------------------
P        = Uinf * ones(Ny, Nx, 'like', xVals);        % Ny×Nx
P(:,1)   = Uinf;                                     % column 1

% We only need to march in **x** now
for i = 2:Nx
    insideNow   =  inMask(:, i);          % Ny×1
    insidePrev  =  inMask(:, i-1);        % Ny×1
    robotNow    =  idxMat(:, i);          % Ny×1 (uint16)
    robotPrev   =  idxMat(:, i-1);
    Pprev       =  P(:, i-1);             % Ny×1

    % CASE 1 : just entered a circle (outside → inside)
    justEntered =  insideNow  & ~insidePrev;
    P(justEntered, i) = max(minPowerX, (1-percentDrop) .* Pprev(justEntered));

    % CASE 2 : staying inside same robot
    staySame    =  insideNow  &  insidePrev & (robotNow == robotPrev);
    P(staySame, i) = Pprev(staySame);

    % CASE 3 : jumped to a different robot (inside → inside, diff idx)
    switchRobot =  insideNow  &  insidePrev & (robotNow ~= robotPrev);
    P(switchRobot, i) = max(minPowerX, (1-percentDrop) .* Pprev(switchRobot));

    % CASE 4 : just exited a circle (inside → outside)
    justExit    = ~insideNow  &  insidePrev;
    P(justExit,  i) = Pprev(justExit);

    % CASE 5 : outside → outside  ⇒ exponential recovery
    stillOut    = ~insideNow  & ~insidePrev;
    gap         = Uinf - Pprev(stillOut);
    P(stillOut, i) = min(Uinf, Pprev(stillOut) + gap*recoveryRate*dx);
end
% -- SMOOTHING (same kernels, now one GPU call) ---------------------------
% Kx = 2*floor(Nx/xSmoothing1)+1;
% Ky = 2*floor(Ny/ySmoothing1)+1;
% kernel = ones(Ky, Kx, 'like', P) / (Kx*Ky);          % Ny-first conv2 style
% Ppad   = padarray(P, floor(size(kernel)/2), 'replicate', 'both');
% Psm    = conv2(Ppad, kernel, 'valid');               % keeps Ny×Nx
% powerVals = Psm;

% Define smoothing sizes and kernel size
Kx = 2*floor(Nx/xSmoothing1) + 1;
Ky = 2*floor(Ny/ySmoothing1) + 1;

% Generate exponential kernel (Ny-first: rows=Y, cols=X)
[xg, yg] = meshgrid(1:Kx, 1:Ky);
cx = ceil(Kx/2);   % center x
cy = ceil(Ky/2);   % center y

% Compute distance from center
ddx = abs(xg - cx);
ddy = abs(yg - cy);

% Exponential decay kernel (can tweak decay rate with alpha)

kernel = exp( -alpha * (ddx + ddy) );
kernel = kernel / sum(kernel(:));    % normalize to sum to 1

% Cast kernel to same type as P (double, single, gpuArray, etc.)
kernel = cast(kernel, 'like', P);

% Apply padded convolution (Ny × Nx remains preserved)
Ppad   = padarray(P, floor(size(kernel)/2), 'replicate', 'both');
Psm    = conv2(Ppad, kernel, 'valid');  % same output size




% --- constants reused in vector form ---------------------------------
deltaOK  = 1;                 % same as before
thrOK    = Uinf - deltaOK;    % free-stream threshold

% --- Boolean mask: where power is already near free-stream -----------
okMask   = Psm >= thrOK;          % Ny × Nx   (gpuArray if useGPU)

% ----- INPUT -----------------------------------------------------------
% okMask : Ny × Nx logical (gpuArray or normal array)
%          1 → “OK”,   0 → “not-OK / barrier”

rowIdx   = (1:Ny).';                          % column-vector of row numbers
if isa(okMask,'gpuArray')
    rowIdx = gpuArray(rowIdx);               % keep everything on the GPU
end

% ----- DISTANCE TO PREVIOUS 0  (looking UP) ----------------------------
% prevZero(row,col) = row-index of the most recent 0 above *or this row*
prevZero = cummax( rowIdx .* (okMask), 1 );  % cummax along columns
distUp   = ~okMask .* ( rowIdx - prevZero );   % consecutive 1`s upward

% ----- DISTANCE TO NEXT 0  (looking DOWN) ------------------------------
okFlip   = flipud(okMask);                    % look from the bottom
nextZero = cummax( rowIdx .* (okFlip), 1 );  % last 0 *below* in original
distDown = ~okMask .* flipud( rowIdx - nextZero );   % consecutive 1~s downward


% distUp(j,i) = rows above j until first OK
% distDown(j,i) = rows below j until first OK
kernelHalfY = min(distUp, distDown);     % Ny × Nx
wallScale = 2*(tanh(3*(2*kernelHalfY  *  dy/maxWallSpan  -  1))+1);
%wallScale = 1;
    powerDef = (Psm/100).^wallScale;
%powerDef = 1;
Psm = max(minPowerY,Psm.*powerDef);



% ... Now you can do your final smoothing ...
% averaging
% Ky = 2*floor(Ny/ySmoothing2)+1;
% Kx = 2*floor(Nx/xSmoothing2)+1;
% kernel = ones(Kx,Ky) / (Kx*Ky);
% padSize = floor(size(kernel)/2); % e.g. [1 1] for a 3x3 kernel
% 
% % Replicate-padding
% powerValsPadded = padarray(powerValsDroped, padSize, 'replicate', 'both');
% 
% powerValsSmoothed = conv2(powerValsPadded, kernel, 'valid');

Kx = 2*floor(Nx/xSmoothing2) + 1;
Ky = 2*floor(Ny/ySmoothing2) + 1;

% Generate exponential kernel (Ny-first: rows=Y, cols=X)
[xg, yg] = meshgrid(1:Kx, 1:Ky);
cx = ceil(Kx/2);   % center x
cy = ceil(Ky/2);   % center y

% Compute distance from center
ddx = abs(xg - cx);
ddy = abs(yg - cy);

% Exponential decay kernel (can tweak decay rate with alpha)

kernel = exp( -beta * (ddx + ddy) );
kernel = kernel / sum(kernel(:));    % normalize to sum to 1

% Cast kernel to same type as P (double, single, gpuArray, etc.)
kernel = cast(kernel, 'like', P);

% Apply padded convolution (Ny × Nx remains preserved)
Ppad   = padarray(Psm, floor(size(kernel)/2), 'replicate', 'both');
powerValsSmoothed    = conv2(Ppad, kernel, 'valid');  % same output size

% Suppose powerVals is Nx x Ny
% Apply convolution, 'same' keeps the output the same size as input

Psm = powerValsSmoothed';
% ---------- gather to CPU if caller expects CPU --------------------------
if useGPU, Psm = gather(Psm); end

end
