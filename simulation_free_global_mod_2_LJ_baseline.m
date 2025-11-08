function [eff, dist_travelled, average_batt, collision_counter] = simulation_free_global_mod_2_LJ_baseline(seed)
%% --- Parameters (tweak here) -------------------------------------------
if nargin > 0, rng(seed,'twister'); end

dt          = 0.5;           % [s]
n_agents    = 20;
robot_rad   = 0.055;         % ~ 11 cm diameter
wind_rad    = 0.15;

% LJ "spacing" parameters
r0          = 0.40;          % desired inter-robot spacing [m]
sigma       = r0 / 2^(1/6);  % standard LJ equilibrium relation
epsilon     = 0.08;          % depth (force scale). Try 0.05–0.15
r_cut       = 1.5*r0;        % cutoff for LJ summation
r_min       = 0.02;          % avoid singularity if r ~ 0

% Alignment and goal terms
R_align     = 0.8;           % neighbor radius for alignment
k_align     = 1.2;           % heading alignment gain (rad/s)
k_goal      = 0.40;          % pull to the left (−x)
k_v         = 4.0;           % force → speed scaling
v_lin_max   = 0.20;          % [m/s]
w_max       = pi/5;          % [rad/s]

% Arena & wind
xRange      = [-5, 5];
yRange      = [-5, 5];
Uinf        = 100;
Nx          = 200; Ny = 200; useGPU = 0;
kappa       = 10;    % drag coefficient
v_wind      = 10;    % [m/s] free-stream

% Battery
P_idle_pct  = 0.10;  % %/s
bat_init    = 100;   % %
bat_scale   = 2.0;   % same scaling you used elsewhere

%% --- Spawn --------------------------------------------------------------
spawn_square_size = 3;     mid = [0,0];
agents = rand(n_agents,4);
agents(:,1) = mid(1) + (agents(:,1)-0.5)*spawn_square_size;
agents(:,2) = mid(2) + (agents(:,2)-0.5)*spawn_square_size;
agents(:,3) = wrapToPi(agents(:,3)*2*pi);
agents(:,4) = bat_init;

% de-clump initial positions
min_dist_initial = 0.1 + 2*robot_rad;
done=false;
while ~done
  done = true;
  for i=1:n_agents
    for j=i+1:n_agents
      d = hypot(agents(i,1)-agents(j,1), agents(i,2)-agents(j,2));
      if d < min_dist_initial
        agents(j,1:2) = mid + (rand(1,2)-0.5)*spawn_square_size;
        done = false;
      end
    end
  end
end

% Walls
wall_left=-5+robot_rad; wall_right=5-robot_rad;
wall_top =5-robot_rad; wall_bottom=-5+robot_rad;
walls=[wall_left,wall_right,wall_top,wall_bottom];

%% --- Viz / video --------------------------------------------------------
v = VideoWriter('baseline_LJ','MPEG-4'); v.Quality=60; v.FrameRate=10; open(v);
figure(1); clf; ax = gca; hold(ax,'on'); axis(ax,'equal'); grid(ax,'on');
xlabel(ax,'X [m]'); ylabel(ax,'Y [m]'); axis(ax,[xRange,yRange]);

%% --- Main loop ----------------------------------------------------------
t=0; collision_counter=0; batteryEmpty=false;
heading_sum=0; step_counter=0;

while ~batteryEmpty
  X = agents(:,1:2);     % positions
  th = agents(:,3);      % headings

  % Pairwise vectors & distances
  % (Compute only once per step for efficiency)
  Dx = X(:,1) - X(:,1).';      % N x N
  Dy = X(:,2) - X(:,2).';
  R2 = Dx.^2 + Dy.^2 + eye(n_agents);  % add eye to avoid 0 on diag
  R  = sqrt(R2);
  ex = Dx ./ R;                % unit vectors i←j (component-wise)
  ey = Dy ./ R;

  % Mask self and beyond cutoff
  mask = (R > r_min) & (R < r_cut);
  % LJ force magnitude from j on i: F_ij = 24eps*(2*(sig/r)^12 - (sig/r)^6)/r  along e_ij
  sig_over_r6  = (sigma.^6) ./ (R.^6 + ~mask);   % safe denom
  sig_over_r12 = sig_over_r6.^2;
  Fmag = 24*epsilon .* (2*sig_over_r12 - sig_over_r6) ./ (R + ~mask);
  Fmag = Fmag .* mask; % apply cutoff and remove self

  % Sum neighbor forces
  Fx_LJ = sum(Fmag .* ex, 2);
  Fy_LJ = sum(Fmag .* ey, 2);

  % Alignment torque (heading to average neighbors within R_align)
  align_mask = (R < R_align) & ~eye(n_agents);
  % average neighbor heading as a unit complex mean
  cmean = sum(align_mask .* cos(th.'), 2) + 1i*sum(align_mask .* sin(th.'), 2);
  th_nb = angle(cmean);              % N×1, NaN if no neighbors
  th_nb(~any(align_mask,2)) = th(~any(align_mask,2));  % if none, keep own
  e_align = wrapToPi(th_nb - th);    % alignment heading error

  % Goal-seeking (left = −x direction)
  Fx_goal = -k_goal * ones(n_agents,1);
  Fy_goal =  0       * ones(n_agents,1);

  % Total "force-like" vector → desired heading
  Fx = Fx_LJ + Fx_goal;
  Fy = Fy_LJ + Fy_goal;

  phi_des = atan2(Fy, Fx);          % world-frame desired bearing
  phi_fwd = th + pi/2;              % your forward direction
  e_heading = wrapToPi(phi_des - phi_fwd);

  % Controls
  v_cmd = v_lin_max * tanh(k_v*sqrt(Fx.^2 + Fy.^2));     % ∈ [0,v_max]
  w_cmd = max(-w_max, min(w_max, 1.6*e_heading + k_align*e_align)); % blend

  vel = [v_cmd, w_cmd];

  % Move + collisions + walls + dynamic window for wind
  [vel_actual, agents, xRange_dyn, collision_counter] = ...
      move_LJ(agents, vel, dt, n_agents, walls, collision_counter);

  % Wind, drag, battery
  [yVals, xVals, powerVals] = ...
      RayTraceCircularRobots(agents, wind_rad, Uinf, xRange_dyn, yRange, Nx, Ny, useGPU);
  F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals, n_agents, vel_actual, v_wind, kappa);
  [agents, ~] = batterydrainage_baseline(agents, vel_actual, F_drag, robot_rad, dt, P_idle_pct, bat_scale);

  % Termination
  batteryEmpty = any(agents(:,4) <= 0);

  % Book-keeping for metrics
  align_step   = mean( cos( agents(:,3) - pi/2 ) );
  heading_sum  = heading_sum + align_step;
  step_counter = step_counter + 1;

  % Draw
  cla(ax);
  plot_wind(ax, yVals, xVals, powerVals);
  plot_agents(ax, agents, robot_rad);
  title(ax, sprintf('LJ Baseline — t = %.1f s', t));
  frame = getframe(gcf); writeVideo(v, frame);
  t = t + dt;
end

%% --- Metrics & return ---------------------------------------------------
avg_heading     = heading_sum / max(1,step_counter);
average_batt    = mean(agents(:,4));
average_x       = mean(agents(:,1));
dist_travelled  = -average_x;                 % left is "good"
collision_time  = collision_counter * dt;

eff = dist_travelled + average_batt/5 - collision_time/250; %#ok<NASGU>

close(v);
end

%% ===== Helpers (kept simple, no quadrants) ==============================
function [vel_actual, agents, xRange, collision_counter] = move_LJ(agents, vel, dt, n_agents, walls, collision_counter)
vel_actual = [vel, agents(:,3)];
theta = agents(:,3);
dx = -vel(:,1).*dt.*sin(theta);
dy =  vel(:,1).*dt.*cos(theta);

agents_old = agents;
agents(:,1) = agents(:,1) + dx;
agents(:,2) = agents(:,2) + dy;
agents(:,3) = wrapToPi(agents(:,3) + vel(:,2)*dt);

% simple collision counting (pairs closer than 2 radii)
D = pdist2(agents(:,1:2), agents(:,1:2));
min_dist = 0.01 + 2*0.055;
num_pairs = nnz(triu((D < min_dist),1));
collision_counter = collision_counter + num_pairs;

% walls (soft clamp + penalize near hits)
wall_margin = 0.055 * 0.5;
wall_hits_step = sum(agents(:,1) > walls(2)-wall_margin | ...
                     agents(:,2) > walls(3)-wall_margin | ...
                     agents(:,2) < walls(4)+wall_margin );
collision_counter = collision_counter + 3*wall_hits_step;

% dynamic x window for wind field
min_x = min(agents(:,1));
max_x = min(max(agents(:,1)), min_x+9.8);
width = max_x - min_x; widening = (10-width)/2;
xRange = [min_x-widening, max_x+widening];

% clamp to walls
agents(:,1) = min(agents(:,1), max_x);
agents(:,2) = min(agents(:,2), walls(3));
agents(:,2) = max(agents(:,2), walls(4));

% reconstruct actual velocity
x_old = agents_old(:,1); y_old = agents_old(:,2);
x_new = agents(:,1);     y_new = agents(:,2);
dist  = hypot(x_new - x_old, y_new - y_old);
vel_actual(:,3) = atan2((y_new - y_old), (x_new - x_old)) - pi/2;
vel_actual(:,1) = dist / dt;
vel_actual(isnan(vel_actual(:,3)),3) = 0;
end

function [agents, batt_drain] = batterydrainage_baseline(agents, vel_actual, F_drag, robot_rad, dt, P_idle_pct, scale)
% velocity vector in world frame
vel_vec = [ -vel_actual(:,1).*sin(vel_actual(:,3)), ...
             vel_actual(:,1).*cos(vel_actual(:,3)) ];
work_wind = sum(vel_vec .* F_drag, 2);  % power taken/given by wind (W proxy)

dv = vel_actual(:,2) * robot_rad;
wheel = [vel_actual(:,1)-dv, vel_actual(:,1)+dv];
P_mot = sum(abs(wheel),2)/4;       % crude motor power proxy

P_use = max(P_idle_pct, P_mot - work_wind);    % %/s proxy
batt_drain = P_use * dt;
agents(:,4) = agents(:,4) - scale * batt_drain;  % apply scale like your code
end

function plot_wind(ax, yVals, xVals, powerVals)
[X,Y] = meshgrid(xVals,yVals);
pcolor(ax, X, Y, powerVals'); shading interp;
c = colorbar; c.Label.String='Windspeed %'; c.Label.FontWeight='bold';
clim([0, max(powerVals(:))]);
end

function plot_agents(ax, agents, r)
for i=1:size(agents,1)
  x=agents(i,1); y=agents(i,2); th=agents(i,3); bat=agents(i,4);
  nb = max(0,min(1,bat/100)); col=[1-nb, nb, 0];  % red→green
  ang = linspace(0,2*pi,50);
  fill(ax, x+r*cos(ang), y+r*sin(ang), col, 'EdgeColor','k');
  quiver(ax, x, y, -0.3*sin(th), 0.3*cos(th), 0, 'Color','r','LineWidth',1);
end
end
