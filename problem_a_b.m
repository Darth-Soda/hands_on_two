%% ========================================================================
%  Accelerometer tracking: (a) plain double integration
%                          (b) Kalman filter with zero-velocity updates (ZUPT)
%  mountType = 'foot' : ZUPT whenever the foot is flat on the ground (stance)
%  mountType = 'hand' : no stance phases exist while walking, so ZUPT is only
%                       applied in the still periods at the start and end
%  ========================================================================
clear; clc; close all;

%% 0. User settings (tune these)
fileName     = 'accel_data.csv';
mountType    = 'hand';   % 'foot' or 'hand'
calibTime    = 1.0;      % [s]  still period at the start -> gravity/bias estimate
normalizeToG = true;     % rescale so the static reading equals 9.81 m/s^2
useSmoother  = true;     % backward (RTS) pass: spreads end-of-walk ZUPT over the whole path
sigma_a      = 0.5;      % [m/s^2] process noise
sigma_z      = 0.01;     % [m/s]   ZUPT measurement noise

if strcmp(mountType, 'foot')
    winTime = 0.10; varThresh = 0.5;  magTol = 0.8;   % stance detector (foot)
else
    winTime = 0.50; varThresh = 0.02; magTol = 0.5;   % "standing still" detector (hand)
end

%% 1. Load data
data = readtable(fileName);
t  = data.time;
t  = t - t(1);          % start time at 0
% t = t/1000;           % uncomment if time is in milliseconds
a_raw = [data.ax data.ay data.az];   % N x 3
% a_raw = a_raw * 9.81; % uncomment if your sensor outputs in g

N  = length(t);
fs = 1/median(diff(t));
fprintf('Samples: %d | duration: %.2f s | fs ~ %.1f Hz\n', N, t(end), fs);

%% 2. Gravity / bias removal (same for BOTH pipelines)
nCal = find(t >= calibTime, 1);
if isempty(nCal), nCal = N; end
g_vec = mean(a_raw(1:nCal, :), 1);
if normalizeToG
    s = 9.81 / norm(g_vec);          % static |a| should be 9.81 m/s^2
    a_raw = a_raw * s;  g_vec = g_vec * s;
    fprintf('Scale correction factor applied: %.3f\n', s);
end
g_mag = norm(g_vec);
a     = a_raw - g_vec;
fprintf('Gravity vector: [%.3f %.3f %.3f], |g| = %.3f\n', g_vec, g_mag);

%% 3. (a) Plain double integration (trapezoidal rule)
v = zeros(N, 3);
p = zeros(N, 3);
for i = 2:N
    dt = t(i) - t(i-1);
    v(i,:) = v(i-1,:) + 0.5 * (a(i,:) + a(i-1,:)) * dt;
    p(i,:) = p(i-1,:) + 0.5 * (v(i,:) + v(i-1,:)) * dt;
end

%% 4. Stationary detection (accelerometer only)
win        = max(3, round(winTime * fs));
mag        = vecnorm(a_raw, 2, 2);
magVar     = movvar(mag, win);
magMean    = movmean(mag, win);
stationary = (magVar < varThresh) & (abs(magMean - g_mag) < magTol);

if strcmp(mountType, 'hand')
    % keep only the leading and trailing still periods
    zuptMask = false(N, 1);
    kFirst = find(~stationary, 1, 'first');
    kLast  = find(~stationary, 1, 'last');
    if isempty(kFirst)
        zuptMask(:) = true;
    else
        zuptMask(1:kFirst-1)  = true;
        zuptMask(kLast+1:end) = true;
    end
else
    zuptMask = stationary;
end
fprintf('ZUPT applied on %.1f %% of samples\n', 100 * mean(zuptMask));

%% 5. (b) Kalman filter + ZUPT (forward pass, optional RTS smoother)
% State x = [px py pz vx vy vz]'. Predict with measured acceleration as input;
% update with pseudo-measurement v = 0 where zuptMask is true.
H  = [zeros(3), eye(3)];
R  = sigma_z^2 * eye(3);
I6 = eye(6);

x0 = zeros(6, 1);
P0 = 1e-6 * eye(6);
xf = zeros(6, N);   Pf = zeros(6, 6, N);   % filtered
xp = zeros(6, N);   Pp = zeros(6, 6, N);   % predicted
Fk = zeros(6, 6, N);
xf(:,1) = x0; Pf(:,:,1) = P0; xp(:,1) = x0; Pp(:,:,1) = P0; Fk(:,:,1) = I6;

x = x0; P = P0;
for i = 2:N
    dt = t(i) - t(i-1);
    u  = 0.5 * (a(i,:) + a(i-1,:))';
    F  = [eye(3), dt*eye(3); zeros(3), eye(3)];
    B  = [0.5*dt^2 * eye(3); dt * eye(3)];
    Q  = sigma_a^2 * [dt^4/4*eye(3), dt^3/2*eye(3);
                      dt^3/2*eye(3), dt^2  *eye(3)];

    x = F * x + B * u;                 % predict
    P = F * P * F' + Q;
    xp(:,i) = x;  Pp(:,:,i) = P;  Fk(:,:,i) = F;

    if zuptMask(i)                     % ZUPT update
        S = H * P * H' + R;
        K = (P * H') / S;
        x = x + K * (0 - H * x);
        P = (I6 - K*H) * P * (I6 - K*H)' + K * R * K';
    end
    xf(:,i) = x;  Pf(:,:,i) = P;
end

xs = xf;
if useSmoother
    for i = N-1:-1:1
        C = Pf(:,:,i) * Fk(:,:,i+1)' / Pp(:,:,i+1);
        xs(:,i) = xf(:,i) + C * (xs(:,i+1) - xp(:,i+1));
    end
end
p_kf = xs(1:3, :)';

%% 6. Side-by-side 3D comparison
figure('Name', '3D position comparison', 'Position', [100 100 1200 520]);
ax1 = subplot(1,2,1);
plotPath(p,    sprintf('(a) Double integration only\nend-point distance = %.1f m', norm(p(end,:))));
ax2 = subplot(1,2,2);
plotPath(p_kf, sprintf('(b) Kalman filter + ZUPT\nend-point distance = %.1f m', norm(p_kf(end,:))));
hlink = linkprop([ax1 ax2], {'CameraPosition', 'CameraUpVector'}); %#ok<NASGU>

%% 7. Per-axis position vs. time (left: raw, right: Kalman)
figure('Name', 'Position vs time', 'Position', [100 100 1200 700]);
lbl = {'X', 'Y', 'Z'};
for k = 1:3
    subplot(3,2,2*k-1);
    plot(t, p(:,k), 'b', 'LineWidth', 1.2); grid on; ylabel([lbl{k} ' (m)']);
    if k == 1, title('(a) Double integration only'); end
    if k == 3, xlabel('Time (s)'); end
    subplot(3,2,2*k);
    plot(t, p_kf(:,k), 'r', 'LineWidth', 1.2); grid on; ylabel([lbl{k} ' (m)']);
    if k == 1, title('(b) Kalman filter + ZUPT'); end
    if k == 3, xlabel('Time (s)'); end
end

%% 8. ZUPT mask check
figure('Name', 'ZUPT detection', 'Position', [100 100 1000 400]);
plot(t, mag, 'k'); hold on;
yl = ylim;
area(t, zuptMask * yl(2), 'FaceColor', 'g', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
yline(g_mag, 'r--');
grid on; xlabel('Time (s)'); ylabel('|a| (m/s^2)');
title('Samples where ZUPT is applied (green)');
legend('|a|', 'ZUPT active', '|g| estimate');

%% 9. Local function
function plotPath(P, ttl)
    plot3(P(:,1), P(:,2), P(:,3), 'LineWidth', 1.5); hold on;
    plot3(P(1,1),   P(1,2),   P(1,3),   'go', 'MarkerFaceColor', 'g');
    plot3(P(end,1), P(end,2), P(end,3), 'rs', 'MarkerFaceColor', 'r');
    grid on; axis equal;
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title(ttl);
    legend('Path', 'Start', 'End', 'Location', 'best');
end