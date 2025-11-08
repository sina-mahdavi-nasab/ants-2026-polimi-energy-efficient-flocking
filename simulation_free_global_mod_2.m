function [eff, dist_travelled, average_batt, collision_counter] = simulation_free_global_mod_2(rules,seed)
%% parameter setting
% rng(1); %set the random seed
    if nargin > 1          % was a 3-rd arg passed in?
            rng(seed,'twister')   % ⇒ set RNG for this one swarm
    end
% Battery drainage has been scaled by 2a
% Idle battery drainage is now 0.1

v = VideoWriter('alone','MPEG-4');   % → MP4, H.264
v.Quality   = 60;   % 0‑100, lower = smaller file (try 50‑70 first)
v.FrameRate = 10;    % fewer fps  ⇒  smaller file; still smooth for swarms
open(v);  




dt = 0.5; %set time-step
n_agents = 5;     %number of agents
robot_rad = 0.055; %diameter = 14 cm???
wind_rad = 0.15; 
xRange = [-5, 5]; %simulation X
yRange = [-5, 5]; %simulation Y
v_max = [0.1 1];
v_avg = 0.1; %velocity all agents are focred to move
v_wind = 10;
t = 0;
collision_counter = 0;


%% GPU
useGPU = 0;

%% wind parameters
% Now proceed with the rest of your RayTrace code
Uinf = 100;             % "full" power
Nx = 200;                % steps in x
Ny = 200;               % steps in y     
kappa = 10;

%% creating box for spawning of agents
spawn_square_size = 3;     %set spawn box size
midpoint = [0,0];          %set spawn box midpoint
min_battery = 100;          %minimum starting battery level
max_battery = 100;         %maximum starting battery level

%% parameters for collision avoidance
% invisible walls:
wall_left = -5+robot_rad;
wall_right = 5-robot_rad;
wall_top = 5-robot_rad;
wall_bottom = -5+robot_rad;
walls = [wall_left, wall_right, wall_top, wall_bottom];
% collision between agents
min_dist = 0.01 + 2*robot_rad; % if agents are within this distance of each other they are moved out of the way

%% spawning the agents     (x, y, heading, batterylevel)
agents = rand(n_agents,4);                                         %creating matrix of random values (between 0 - 1)

agents(:,1) = midpoint(1)+(agents(:,1)-0.5)*spawn_square_size;     %scaling of x_coordinate
agents(:,2) = midpoint(2)+(agents(:,2)-0.5)*spawn_square_size;     %scaling of y_coordinate
agents(:,3) = wrapToPi(agents(:,3)*2*pi);                                    %scaling of heading
agents(1:n_agents-1,4) = max_battery; %scaling of battery
agents(end,4) = min_battery;


min_dist_initial = 0.1 + 2 * robot_rad;
collision_detected = true;
while collision_detected
    collision_detected = false;
    for i = 1:n_agents
        for j = i+1:n_agents
            % Calculate the distance between agents i and j
            distance = sqrt((agents(i, 1) - agents(j, 1))^2 + (agents(i, 2) - agents(j, 2))^2);
            if distance < min_dist_initial  % if too close, reassign one of the agents
                % Reassign the position of agent j
                agents(j, 1) = midpoint(1) + (rand - 0.5) * spawn_square_size;
                agents(j, 2) = midpoint(2) + (rand - 0.5) * spawn_square_size;
                collision_detected = true;  % set flag to true to recheck all agents
            end
        end
    end
end


%% velocity spawning     (velocity, angular velocity)
vel = zeros(n_agents,2); %zero initial velocities

%% the while loop, runs until 1 robot has no battery
batteryEmpty = false;
agents_log = [];
inputs_log = [];
inputs = zeros(10,n_agents);
batt_drain_log = [];
vel_log = [];
heading_sum    = 0;      %  ← NEW
step_counter   = 0;      %  ← NEW
% agents_log = cat(3,agents_log,agents);

inputs_log = cat(3,inputs_log,inputs);
% [yVals, xVals, powerVals] = RayTraceCircularRobots(agents, robot_rad, Uinf, xRange, yRange, Nx, Ny, useGPU);

% comment the next lines to plot / not plot
figure(1); clf;
ax = gca;
hold(ax, 'on');
axis(ax, 'equal');
grid(ax, 'on');
title(ax, ['Swarm intelligence experiment -- t = ', num2str(t)]);
xlabel(ax, 'X - [m]');
ylabel(ax, 'Y - [m]');
axis(ax, [xRange, yRange]);
set(gcf, 'Position', [0 0 5000 8000]);

W = repmat(struct('W1',[],'W2',[],'W3',[]), n_agents, 1);

for i = 1:n_agents
    W(i).W1 = 2*randn(10,10)-1;   % 10x10  uniform in [-1,1]
    W(i).W2 = 2*rand(10,10)-1;   % 10x10  uniform in [-1,1]
    W(i).W3 = 2*rand(10,2)-1 ;   % 10×2   uniform in [-1,1]
end

while ~batteryEmpty
    %inputs order: [top[dist,bear,head,batt] bottom[] right[] left[] battery]
    inputs = getsensordata(agents);
    inputs_log = cat(3,inputs_log,inputs);
    for i = 1:n_agents
        input_i = inputs(:,i);        % 10 × 1 sensory vector

        % forward pass + Hebbian update for THIS agent only
        [vel_i, W1, W2, W3] = hebbianStep(input_i, W(i).W1, W(i).W2, W(i).W3, rules);

        vel(i,:) = vel_i.';           % store desired (v, ω)
        W(i).W1  = W1;                % keep the learned weights
        W(i).W2  = W2;
        W(i).W3  = W3;
    end

    [vel_actual, agents, xRange, collision_counter] = move(agents,vel,dt,v_avg,n_agents,min_dist,walls, collision_counter);                          %calculates the coordinates it will have after the step
    align_step   = mean( cos( agents(:,3) - pi/2 ) ); % swarm‑level score
    heading_sum  = heading_sum + align_step;
    step_counter = step_counter + 1;

    

    vel_log = cat(3,vel_log, vel);
    [yVals, xVals, powerVals] = ...
      RayTraceCircularRobots(agents, wind_rad, Uinf, xRange, yRange, Nx, Ny, useGPU);
    F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals,n_agents,vel_actual,v_wind,kappa);
    % F_drag = zeros(20,2);
    [agents, batt_drain] = batterydrainage(agents,vel_actual,F_drag,robot_rad,dt); %calculates the battery it will lose in the current step
    % batt_drain_log = [batt_drain_log, batt_drain];
    batteryEmpty = any(agents(:, 4) <= 0);                 %checks if anyones battery is empty
    % agents_log = cat(3,agents_log,agents);                 %logs the files

    t = t + dt;
    % comment the next line to plot / not plot
    axis(ax, [xRange, yRange]);
    plot_all(ax,agents,robot_rad,yVals, xVals, powerVals,t,v)
end         
%% Fitness function
%distance = sum(vecnorm(agents(:,1:2)-agents_log(1,1:2,1),2,2));   % total travel
%battUsed = max_battery*n_agents - sum(agents(:,4));               % Wh consumed
avg_heading  = heading_sum / step_counter;
average_batt = mean(agents(:,4));
average_x = mean(agents(:,1));
dist_travelled = -average_x;
% fprintf('the distance travelled = ' + dist_travelled)
collision_time = collision_counter * dt;
eff      = dist_travelled + average_batt/5 - collision_time/250;% + avg_heading + average_batt/2 - collision_time/20 ; % energy-efficiency (higher = better)
% close(v);

% %% plotting the path of all agents
% figure(2); cla;
% ax2 = gca;
% hold(ax2, 'on');
% axis(ax2, 'equal');
% grid(ax2, 'on');
% title(ax2, 'Trajectories of 20 Robots');
% xlabel(ax2, 'X');
% ylabel(ax2, 'Y');
% 
% yRange2 = [-5,5];
% % xRange2 = squeeze([min(min(agents_log(:,1,:)))-1,max(max(agents_log(:,1,:)))+1]);
% xRange2 = squeeze([-35,max(max(agents_log(:,1,:)))+1]);
% axis(ax2, [xRange2, yRange2]);
% % for i = 1:10
% %     x = squeeze(agents_log(i,1,:));
% %     y = squeeze(agents_log(i,2,:));
% %     plot(x, y, 'LineWidth', .2); % Plot each robot's path
% % end
% plot(agents(:,1),agents(:,2),'ko',LineWidth=4,MarkerSize=2.5)
% hold off;
% 
%% plotting the inputs to the neural network over time
% ttt = (1:step_counter)/2;
% for i = 9:9
%     axx = gobjects(6,1);
%     figure(i+11)
%     % plot(ttt,squeeze(inputs_log(:,i,2:end))')
%     axx(1) = subplot(7,1,1);
%     plot(ttt,squeeze(inputs_log(1:2,i,2:end))')
%     legend('distance','bearing',Location='northwest')
%     ylabel('front')
%     axx(2) = subplot(7,1,2);
%     plot(ttt,squeeze(inputs_log(3:4,i,2:end))')
%     legend('distance','bearing' ,Location='northwest')
%     ylabel('back')
%     axx(3) = subplot(7,1,3);
%     plot(ttt,squeeze(inputs_log(5:6,i,2:end))')
%     legend('distance','bearing',Location='northwest')
%     ylabel('right')
%     axx(4) = subplot(7,1,4);
%     plot(ttt,squeeze(inputs_log(7:8,i,2:end))')
%     legend('distance','bearing',Location='northwest')
%     ylabel('left')
%     axx(5) = subplot(7,1,5);
%     plot(ttt,squeeze(inputs_log(9:10,i,2:end))')
%     legend('battery','compass',Location='northwest')
%     % ylabel('back'0)
%     axx(6) = subplot(7,1,6);
%     plot(ttt,squeeze(vel_log(i,:,:)))
%     legend('velocity','\omega',Location='northwest')
%     ylabel('output')
%     axx(7) = subplot(7,1,7);
%     % [~,positions] = sort(squeeze(agents_log(:,1,2:end)),1);
%     % [~, ranks] = sort(positions, 1);
%     % plot(ttt,ranks(i,:))%[squeeze(agents_log(i,1,2:end))-mean(squeeze(agents_log(:,1,2:end)),1)'])
%     % legend('x','y' ,Location='northwest')
%     % ylabel('position')
% 
%     linkaxes(axx, 'x');
%     xlim([0,max(ttt)]);
%     xlabel('time')
% 
%     % figure(i+30)
%     % plot(ttt,squeeze(inputs_log([7,10],i,2:end))')
%     % hold on
%     % plot(ttt,squeeze(vel_log(i,:,:)))
%     % legend('right distance' ,'left distance', 'velocity','\omega')
% end

% %%
% % Assuming agents_log is a 3D matrix of size [n_agents, 4, n_steps]
% % agents_log(i, 1:4, t) = [x, y, heading, battery] for agent i at time t
% 
% n_steps = size(agents_log, 3);
% 
% % Preallocate averages
% avg_front = zeros(n_steps, 1);
% avg_middle = zeros(n_steps, 1);
% avg_back = zeros(n_steps, 1);
% 
% for l = 1:n_steps
%     % Extract x-coordinates and battery levels
%     x_coords = squeeze(agents_log(:, 1, l));    % 20×1 vector
%     battery = squeeze(agents_log(:, 4, l));      % 20×1 vector
% 
%     % Sort agents by x-coordinate (left to right)
%     [~, sort_idx] = sort(-x_coords);
% 
%     % Define group indices
%     group_size = ceil(n_agents / 3);
%     back_idx = sort_idx(1:group_size);           % Leftmost (lowest x)
%     middle_idx = sort_idx(group_size+1:end-group_size);
%     front_idx = sort_idx(end-group_size+1:end);  % Rightmost (highest x)
% 
%     % Compute averages
%     avg_front(l) = mean(battery(front_idx));
%     avg_middle(l) = mean(battery(middle_idx));
%     avg_back(l) = mean(battery(back_idx));
% end
% figure(3);cla;
% time = (0:n_steps-1) * dt;  % Assuming dt is your timestep duration
% plot(time, avg_front,  'LineWidth', 2); hold on;
% plot(time, avg_middle,  'LineWidth', 2);
% plot(time, avg_back,  'LineWidth', 2);
% grid on;
% 
% xlabel('Time');
% ylabel('Average Battery Level');
% title('Battery Level by Position');
% legend('Front', 'Middle', 'Back', 'Location', 'best');
% 
% end
end

%% 1 function for all plotting
function plot_all(ax,agents,r,yVals, xVals, powerVals,t,v)
    % if mod(t,2) < 1e-6 || mod(t,2) > 2-1e-6
        cla; %clears the drawing of last plot
        plot_wind(ax, yVals, xVals, powerVals)
        plot_agents(ax,agents, r);
        title(ax, ['Swarm intelligence experiment -- t = ', num2str(t)]);
        drawnow;
        frame = getframe(gcf);  % Capture the figure frame
        writeVideo(v, frame);
    % end
end
%% plotting
% Create base triangle (equilateral, centered at origin)
    function plot_agents(ax, agents, r)
    % Parameters for drawing
    arrowLen = 0.3;   % length of direction arrow

    % Loop over each agent to plot
    for i = 1:size(agents, 1)
        % Extract position and heading
        x = agents(i, 1);
        y = agents(i, 2);
        theta = agents(i, 3);
        battery_level = agents(i, 4);  % battery level

        % Normalize battery level to [0, 1]
        normalized_battery = (battery_level - 0) / (100 - 0);  % Assuming battery is between 0 and 100

        % Map the normalized battery level to a color (green for full, red for empty)
        color = [min(1,1 - normalized_battery), max(0,normalized_battery), 0];  % RGB: Green (0,1,0) to Red (1,0,0)

        % Plot a circle for each agent
        % ------------------------------------------------------
        ang = linspace(0, 2 * pi, 50);
        xcirc = x + r * cos(ang);
        ycirc = y + r * sin(ang);

        % Plot circle with battery level as color
        fill(ax, xcirc, ycirc, color, 'EdgeColor', 'k');  % Color fill based on battery level

        % Plot an arrow to show heading
        % ------------------------------------------------------
        quiver(ax, x, y, ...
            -arrowLen * sin(theta), arrowLen * cos(theta), ...
            0, ...
            'Color', 'r', 'LineWidth', 1, 'MaxHeadSize', 2);
    end
end

%% function for moving the agents and checking collisions
    function [vel_actual, agents, xRange,collision_counter] = move(agents,vel,dt,v_avg,n_agents,min_dist,walls, collision_counter)
% creating a vector to store the actual velocity and direction to calculate
% the wind force. [velocity, ang. velocity, heading]

vel_actual = [vel, agents(:,3)];
theta = agents(:,3);                    %angle of the agent
dx = -vel(:,1).*dt.*sin(theta);         %calc dx using velocity and heading
dy = vel(:,1).*dt.*cos(theta);          %calc dy using velocity and heading
agents_old = agents;
agents(:,1)=agents(:,1)+dx;
agents(:,2)=agents(:,2)+dy;
agents(:,3)=wrapToPi(agents(:,3)+vel(:,2)*dt);    %updating heading using angular velocity

% Assuming `agents` is a 20×2 matrix where each row is [x, y]
agents_xy = agents(:,1:2);  % Extract x and y coordinates
D = pdist2(agents_xy, agents_xy);  % Compute pairwise distancesf

% Logical matrix: true if distance < d_thresh and not self
close_agents = (D < min_dist) & ~eye(size(D));

% Count for each agent
num_pairs = nnz(triu(close_agents, 1)); % counts each pair once

collision_counter = collision_counter + num_pairs;

wall_margin      = 0.055 * 0.5;   % tiny slack so "near" ≠ "hit"
wall_hits_step   = sum( agents(:,1) >  walls(2) - wall_margin | ... % right
                        agents(:,2) >  walls(3) - wall_margin | ... % top
                        agents(:,2) <  walls(4) + wall_margin );    % bottom

collision_counter  = collision_counter + 3*wall_hits_step;

% checking if any of the agents are within each other or outside the frame
% and moving them

% walls = [wall_left, wall_right, wall_top, wall_bottom];
max_iter = 10;


% establishing order for priority of collisions
corners = [5,5;5,-5;-5,5;-5,-5];
dist_corner1 = vecnorm(agents(:,1:2)-corners(1,:),2,2);
dist_corner2 = vecnorm(agents(:,1:2)-corners(2,:),2,2);
dist_corner3 = vecnorm(agents(:,1:2)-corners(3,:),2,2);
dist_corner4 = vecnorm(agents(:,1:2)-corners(4,:),2,2);
corner_distances = [dist_corner1,dist_corner2,dist_corner3,dist_corner4];
dist_to_corner = min(corner_distances,[],2);
[~,order2] = sort(dist_to_corner);

[~,order1] = sort(-agents(:,1)); % we check the agents from right to left

min_x = min(agents(:,1));
max_x = min(max(agents(:,1)),min_x+9.8);

% max_x = max(agents(:,1));
% min_x = max(min(agents(:,1)),max_x-9.8);

% setting the new range for calculating the wind:
width = max_x - min_x;
widening = (10-width)/2;
xRange = [min_x-widening, max_x+widening];

% for iter = 1:max_iter
%     % moved = false;
%     if iter < max_iter / 2+1
%         order = order1;
%     else
%         order = order2;
%     end
%     for idx = 1:n_agents
%         i = order(idx);
%         % checking if any of the agents are out of the invisible walls and 
%         % moving them if they are outside
% 
%         for jdx = idx+1:n_agents
%             j = order(jdx);
%             if i ~= j
%                 dx = agents(j,1) - agents(i,1);
%                 dy = agents(j,2) - agents(i,2);
%                 angle = atan2(dy, dx);    %angle [-pi, pi] with zero to the right
%                 distance = sqrt((agents(i,1) - agents(j,1))^2 + (agents(i,2) - agents(j,2))^2);
%                 % checking if agent i and j are too close together and moving
%                 % them if they are
%                 if distance < min_dist
%                     agents(j,1) = agents(i,1) + min_dist*cos(angle);
%                     agents(j,2) = agents(i,2) + min_dist*sin(angle);
%                     x_old = agents_old(j,1);
%                     y_old = agents_old(j,2);
%                     x_new = agents(j,1);
%                     y_new = agents(j,2);
%                     dist = sqrt((x_old - x_new)^2+(y_old - y_new)^2);
%                     vel_actual(j,3) = atan2((y_new - y_old),(x_new - x_old))-pi/2;
%                     vel_actual(j,1) = dist / dt;
%                     % 
%                     % collision_counter = collision_counter+1;
%                     % moved = true;
%                 end
%             end
%         end
% 
    agents(:,1) = min(agents(:,1),max_x);
    % agents(:,1) = max(agents(:,1),min_x);
    % agents(:,1) = min(agents(:,1),walls(2));
    agents(:,2) = min(agents(:,2),walls(3));
    agents(:,2) = max(agents(:,2),walls(4));
%     end
%     % if moved == false
%     %     break;
%     % end
% end

% agents(:,1) = max(agents(:,1),min_x);
% % agents(:,1) = min(agents(:,1),walls(2));
% agents(:,2) = min(agents(:,2),walls(3));
% agents(:,2) = max(agents(:,2),walls(4));
x_old = agents_old(:,1);
y_old = agents_old(:,2);
x_new = agents(:,1);
y_new = agents(:,2);
dist = sqrt((x_old - x_new).^2+(y_old - y_new).^2);
vel_actual(:,3) = atan2((y_new - y_old),(x_new - x_old))-pi/2;
vel_actual(:,1) = dist / dt;

for i = 1:n_agents
    if isnan(vel_actual(i,3))
        vel_actual(i,3) = 0;
    end

    % % add v_avg velocity to the velocity vector
    % v_x = -vel_actual(i,1)*sin(vel_actual(i,3))-v_avg;
    % v_y = vel_actual(i,1)*cos(vel_actual(i,3));
    % vel_actual(i,1) = sqrt(v_x^2+v_y^2);
    % vel_actual(i,3) = wrapToPi(atan2(v_y,v_x)-pi/2);

    % if vel_actual(i,3) < -pi
    %     vel_actual(i,3) = vel_actual(i,3)+2*pi;
    % elseif vel_actual(i,3) > pi
    %     vel_actual(i,3) = vel_actual(i,3)-2*pi;
    % end
end

end
%% function for calculating the batterydrainage
function [agents,batt_drain] = batterydrainage(agents,vel_actual,F_drag,robot_rad,dt)
vel_vec = zeros(length(agents(:,1)),2);     %to create a vector of the velocities to calculate the work of the wind force
vel_vec(:,1) = -vel_actual(:,1).*sin(vel_actual(:,3));
vel_vec(:,2) = vel_actual(:,1).*cos(vel_actual(:,3));
dottprod = zeros(length(agents(:,1)),1);
for i = 1:length(agents(:,1))
    dottprod(i) = vel_vec(i,:)*F_drag(i,:)';       %calculates the work done by the wind
end
dv = vel_actual(:,2)*robot_rad; %difference in velocity between right and left wheel
wheels_vel = [vel_actual(:,1)-dv,vel_actual(:,1)+dv]; %first column left wheel velocity, 2nd right wheel
%We need to double check the formula for battery drainage, how does the
%battery drainage depend on the velocity (not regarding wind)
batt_drain = max(sum(abs(wheels_vel),2)/4-dottprod,0.10)*dt ;           %power of velocity - work by wind, max to have a minimum power usage when not moving or moving with the wind
% / scaling for batterysize
agents(:,4) = agents(:,4)-2*batt_drain(:);    %update battery level

end

%% get sensory data for inputs of NN
function inputs = getsensordata(agents)
n_agents = length(agents(:,1));
% Initialize matrices to store closest distances in each quadrant
front_distances = zeros(n_agents, 1);
back_distances = zeros(n_agents, 1);
right_distances = zeros(n_agents, 1);
left_distances = zeros(n_agents, 1);
% bearing initialization
front_bearing = zeros(n_agents, 1);
back_bearing = zeros(n_agents, 1);
right_bearing = zeros(n_agents, 1);
left_bearing = zeros(n_agents, 1);
% heading initialization
front_heading = zeros(n_agents, 1);
back_heading = zeros(n_agents, 1);
right_heading = zeros(n_agents, 1);
left_heading = zeros(n_agents, 1);
% Set default distances
front_distances(:) = 2.01;
back_distances(:) = 2.01;
right_distances(:) = 2.01;
left_distances(:) = 2.01;
% Set default battery levels
front_battery = zeros(n_agents, 1);
back_battery = zeros(n_agents, 1);
right_battery = zeros(n_agents, 1);
left_battery = zeros(n_agents, 1);

for i = 1:n_agents
    for j = 1:n_agents
        if i ~= j
            dx = agents(j,1) - agents(i,1);
            dy = agents(j,2) - agents(i,2);
            angle = atan2(dy, dx);    %angle [-pi, pi] with zero to the right
            rel_angle = angle - pi/2 - agents(i,3);
            if rel_angle > pi
                rel_angle = rel_angle - 2*pi;
            elseif rel_angle < -pi
                rel_angle = rel_angle + 2*pi;
            end
            distance = sqrt((agents(i,1) - agents(j,1))^2 + (agents(i,2) - agents(j,2))^2);
            if (rel_angle >= -3*pi/4 && rel_angle <= -pi/4) % Right quadrant
                if distance < right_distances(i)
                    right_distances(i) = distance;
                    right_bearing(i) = rel_angle+3*pi/4;
                    right_heading(i) = agents(j,3);
                    right_battery(i) = agents(j,4);
                end
            elseif (rel_angle >= -pi/4 && rel_angle <= pi/4) % Top quadrant
                if distance < front_distances(i)
                    front_distances(i) = distance;
                    front_bearing(i) = rel_angle+pi/4;
                    front_heading(i) = agents(j,3);
                    front_battery(i) = agents(j,4);
                end
            elseif (rel_angle <= -3*pi/4 || rel_angle >= 3*pi/4) % Left quadrant
                if distance < back_distances(i)
                    back_distances(i) = distance;
                    back_heading(i) = agents(j,3);
                    back_battery(i) = agents(j,4);
                    if rel_angle <= 0
                        back_bearing(i) = rel_angle+5*pi/4;
                    elseif rel_angle > 0
                        back_bearing(i) = rel_angle-3*pi/4;
                    end
                end
            elseif (rel_angle >= pi/4 && rel_angle <= 3*pi/4) % Bottom quadrant
                if distance < left_distances(i)
                    left_distances(i) = distance;
                    left_bearing(i) = rel_angle-pi/4;
                    left_heading(i) = agents(j,3);
                    left_battery(i) = agents(j,4);
                end
            end

        end
    end
end

inputs = [front_distances*2/2.01-1, front_bearing*4/pi-1, ...
 back_distances*2/2.01-1, back_bearing*4/pi-1, ...
 right_distances*2/2.01-1, right_bearing*4/pi-1,  ...
left_distances*2/2.01-1, left_bearing*4/pi-1, ...
 agents(:,4)/50-1, agents(:,3)/pi];
inputs = inputs';

end

%% function to plot the wind
function plot_wind(ax, yVals, xVals, powerVals)
% Create meshgrid for plotting
[X, Y] = meshgrid(xVals, yVals);

pcolor(ax ,X, Y, powerVals');
shading interp;
c = colorbar;
c.Label.String = 'Windspeed %';
c.Label.FontSize = 12;
c.Label.FontWeight = 'bold';
clim([0, max(powerVals(:))]);  % scale color

end



%% Wind force on agents:
function F_drag = dragforce(agents, wind_rad, xVals, yVals, powerVals,n_agents,vel_actual,v_wind,kappa)
    % for some reason powerVals is flipped
    powerVs = powerVals';
    % powerVals = flipud(powerVals);
    % make an initial matrix of 100's which will be updated with each
    % agents powerVal
    powerVals_agents = 100*ones(n_agents,1);
    % dragforce matrix, first column will be updated with the x-value of
    % the drag force, the second column is 0 --> each row is now a vector
    % of the drag force
    F_drag = zeros(n_agents,2);

    %we want to find the gridpoint just in front of the agent and then we
    %want to take its powerVal
    for i = 1:n_agents
        %get the coordinate of agent i
        x_r = agents(i,1);
        y_r = agents(i,2);
        % count the number of gridpoints to the left of the agent
        x_to_left = find(xVals <= x_r-1.1*wind_rad);
        % the number of gridpoints to the left of the agent
        % x = x_to_left(end);

        if ~isempty(x_to_left)
            x = x_to_left(end);
        else
            % Handle the case where x_to_left is empty
            % For example, set x to a default value or throw a custom error
            x = 0; % or whatever is appropriate
        end

        if x < 3
            % if the agent is all the way to the left the agent gets a
            % lower powerVal for some reason so we set it to a realistic
            % value
            powerVals_agents(i) = 100;
        else
            %find the closest y-grid point
            [~,y] = min(abs(yVals - y_r));
            % take the powervalue of the grid point
            powerVals_agents(i) = powerVs(y,x);
        end

        % convert the powerVal to wind velocity
        v_wind_agent = powerVals_agents(i)/100 * v_wind; 
        % find the paralel velocity of the agent to the wind
        v_parallel = vel_actual(i,1)*sin(vel_actual(i,3));
        % calculate the relative wind velocity
        v_rel = v_wind_agent+v_parallel;
        % calculate the drag force of the wind
        F_drag(i,1) = 1/2 * 1.225 * 0.0045 * kappa * v_rel^2;

    end
end



