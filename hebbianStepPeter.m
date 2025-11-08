% hebbianStep  —  Forward pass + Hebbian weight update for one robot
%                 (version with TWO hidden layers)
function [vout, W1, W2, W3] = hebbianStep(in, W1, W2, W3, R)
    %% Forward ----------------------------------------------------------
    h1  = relu(in' * W1);          % in  (13×1) → h1 (1×13)
    h2  = relu(h1  * W2);          % h1 (1×13) → h2 (1×13)
    out = tanh(h2  * W3);          % h2 (1×13) → out(1×2)  [v  w]
    vout = [out(1) * 0.2 , out(2) * pi/5];

    %% Local Hebbian update -------------------------------------------
    eta = 0.1;

    W1 = W1 + eta * ( R.A1 .* (in  * h1) + R.B1 .* repmat(in ,1,10) + ...
                      R.C1 .* repmat(h1,10,1) + R.D1 );

    W2 = W2 + eta * ( R.A2 .* (h1' * h2) + R.B2 .* repmat(h1',1,10) + ...
                      R.C2 .* repmat(h2,10,1) + R.D2 );

    W3 = W3 + eta * ( R.A3 .* (h2' * out) + R.B3 .* repmat(h2',1,2) + ...
                      R.C3 .* repmat(out,10,1) + R.D3 );

    %% Normalise weights ----------------------------------------------
    W1 = normalize(W1);
    W2 = normalize(W2);
    W3 = normalize(W3);
end

%% -----------------------  Basic helpers  -----------------------------
function y = relu(x)
    y = max(0, x);
end

function Wn = normalize(W)
    maxval = max(abs(W(:)));
    if maxval > 1
        Wn = W / maxval;
    else
        Wn = W;
    end
end
