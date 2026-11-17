%Code for problem C
clear; clc; close all;

%% 1. Load Dataset & Remove Pre-Calibration / Summary Rows
raw = readtable('CST03_left.csv', 'VariableNamingRule', 'preserve');
idx_cal_end = find(raw{:, 1} == -1, 1, 'first');
data = raw(idx_cal_end + 2 : end - 1, :);   % Clean continuous trial (17,079 samples)

t  = data.time - data.time(1);
dt = [median(diff(data.time)); diff(t)];
dt(dt <= 0) = median(dt(dt > 0));
N  = length(t);

G      = 9.80665;                           % Standard gravity (m/s^2)
acc_b  = [data.ax, data.ay, data.az] * G;   % Convert g -> m/s^2
gyro_b = deg2rad([data.gx, data.gy, data.gz]); % Convert deg/s -> rad/s

%% 2. Static Initialization & Sensor Bias Calibration (First 2.0 s = 200 samples)
cal_n     = 200;
acc_init  = mean(acc_b(1:cal_n, :), 1);
gyro_bias = mean(gyro_b(1:cal_n, :), 1);

% Scale factor normalization so static gravity magnitude equals G
acc_b    = acc_b * (G / norm(acc_init));
acc_init = mean(acc_b(1:cal_n, :), 1);

% Initial Body-to-World rotation matrix R_init (World Z = Up) via Rodrigues' formula
g_unit = acc_init(:) / norm(acc_init);
z_w    = [0; 0; 1];
v      = cross(g_unit, z_w);
s      = norm(v);
c      = dot(g_unit, z_w);
vx     = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
R_init = eye(3) + vx + (vx * vx) * ((1 - c) / (s^2 + 1e-12));

%% 3. Baseline: Unfiltered Double Integration (No Kalman Filter)
acc_w_fixed = (R_init * acc_b')' - repmat([0, 0, G], N, 1);
vel_unfilt  = cumtrapz(t, acc_w_fixed);
pos_unfilt  = cumtrapz(t, vel_unfilt);

%% 4. Approach 1: Accelerometer-Only Kalman Filter with ZUPT
acc_mag  = vecnorm(acc_b, 2, 2);
acc_std  = movstd(acc_mag, 5);
zupt_acc = acc_std < 2.0;                   % Stance detection using accel variance

x_acc = zeros(6, N);                        % State: [px; py; pz; vx; vy; vz]
P1    = eye(6) * 1e-4;
H     = [zeros(3, 3), eye(3)];
R_z   = eye(3) * (0.01^2);                  % ZUPT velocity measurement noise
q_acc = 0.5^2;                              % Process acceleration noise variance

for k = 2:N
    d_t = dt(k);
    F = [eye(3), d_t * eye(3); zeros(3), eye(3)];
    B = [0.5 * d_t^2 * eye(3); d_t * eye(3)];
    Q = [0.25 * d_t^4 * eye(3), 0.5 * d_t^3 * eye(3); ...
         0.5 * d_t^3 * eye(3),  d_t^2 * eye(3)] * q_acc;

    % Prediction step
    x_pred = F * x_acc(:, k - 1) + B * acc_w_fixed(k, :)';
    P_pred = F * P1 * F' + Q;

    % Zero-Velocity Update (ZUPT) during stance
    if zupt_acc(k)
        y   = -H * x_pred;
        S   = H * P_pred * H' + R_z;
        K   = (P_pred * H') / S;
        IKH = eye(6) - K * H;
        x_acc(:, k) = x_pred + K * y;
        P1 = IKH * P_pred * IKH' + K * R_z * K';
    else
        x_acc(:, k) = x_pred;
        P1 = P_pred;
    end
end

%% 5. Approach 2: Sensor Fusion (Gyro Attitude + Accel Gravity Correction + ZUPT KF)
gyro_rms   = sqrt(movmean(sum((gyro_b - gyro_bias).^2, 2), 5));
zupt_fused = (gyro_rms < 1.2) & (acc_std < 2.0);

R       = R_init;
x_fus   = zeros(6, N);
P2      = eye(6) * 1e-4;
yaw_deg = zeros(N, 1);

for k = 2:N
    d_t = dt(k);
    w   = (gyro_b(k, :) - gyro_bias)';
    ang = norm(w) * d_t;

    % Attitude integration on SO(3) via exponential map
    if ang > 1e-12
        ax = w / norm(w);
        Kw = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
        R  = R * (eye(3) + sin(ang) * Kw + (1 - cos(ang)) * (Kw * Kw));
    end

    % Accelerometer gravity tilt correction during stance
    if zupt_fused(k)
        a_u  = acc_b(k, :)' / norm(acc_b(k, :));
        g_eb = R' * z_w;
        corr = 0.05 * cross(a_u, g_eb);
        Kc   = [0 -corr(3) corr(2); corr(3) 0 -corr(1); -corr(2) corr(1) 0];
        R    = R * (eye(3) + Kc);
        [U, ~, Vt] = svd(R);
        R    = U * Vt';                      % Re-orthogonalize R onto SO(3)
    end

    yaw_deg(k) = rad2deg(atan2(R(2, 3), R(1, 3)));
    a_w = R * acc_b(k, :)' - [0; 0; G];

    F = [eye(3), d_t * eye(3); zeros(3), eye(3)];
    B = [0.5 * d_t^2 * eye(3); d_t * eye(3)];
    Q = [0.25 * d_t^4 * eye(3), 0.5 * d_t^3 * eye(3); ...
         0.5 * d_t^3 * eye(3),  d_t^2 * eye(3)] * q_acc;

    x_pred = F * x_fus(:, k - 1) + B * a_w;
    P_pred = F * P2 * F' + Q;

    if zupt_fused(k)
        y   = -H * x_pred;
        S   = H * P_pred * H' + R_z;
        K   = (P_pred * H') / S;
        IKH = eye(6) - K * H;
        x_fus(:, k) = x_pred + K * y;
        P2 = IKH * P_pred * IKH' + K * R_z * K';
    else
        x_fus(:, k) = x_pred;
        P2 = P_pred;
    end
end

%% 6. Print Summary Metrics & Generate Comparison Plots
path_len = sum(vecnorm(diff(x_fus(1:2, :), 1, 2), 2, 1));
fprintf('Final Unfiltered Position (m): [%.2f, %.2f, %.2f]\n', pos_unfilt(end, :));
fprintf('Final Accel-Only KF Pos (m):   [%.2f, %.2f, %.2f]\n', x_acc(1:3, end));
fprintf('Final Fused KF Position (m):   [%.2f, %.2f, %.2f]\n', x_fus(1:3, end));
fprintf('Total Horizontal Path (m):     %.2f m\n', path_len);

figure('Position', [100, 100, 1250, 440]);

subplot(1, 3, 1);
plot(x_acc(1, :), x_acc(2, :), 'r--', 'LineWidth', 1.4); hold on;
plot(x_fus(1, :), x_fus(2, :), 'b-', 'LineWidth', 1.8);
plot(x_fus(1, 1), x_fus(2, 1), 'go', 'MarkerFaceColor', 'g', 'MarkerSize', 7);
plot(x_fus(1, end), x_fus(2, end), 'kx', 'MarkerSize', 9, 'LineWidth', 2);
grid on; axis equal;
xlabel('X Position (m)'); ylabel('Y Position (m)');
title('Part (c): 2D Trajectory Comparison');
legend('Approach 1: Accel-Only KF', 'Approach 2: Sensor Fusion KF', 'Start', 'End', 'Location', 'best');

subplot(1, 3, 2);
plot(t, x_fus(3, :), 'm-', 'LineWidth', 1.8);
grid on;
xlabel('Time (s)'); ylabel('Elevation Z (m)');
title('Elevation Z(t): Walking Path Profile');

subplot(1, 3, 3);
semilogy(t, vecnorm(pos_unfilt, 2, 2), 'k-', 'LineWidth', 1.5); hold on;
semilogy(t, vecnorm(x_fus(1:3, :)', 2, 2), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)'); ylabel('Displacement Magnitude (m, log)');
title('Drift: Unfiltered vs. Kalman Filter');
legend('Unfiltered Double Integration', 'Fused Kalman Filter (ZUPT)', 'Location', 'best');