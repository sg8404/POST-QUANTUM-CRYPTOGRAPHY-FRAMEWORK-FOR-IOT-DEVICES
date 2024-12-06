% Parameters
voltage = 3.3;          % Voltage in Volts
current_mc = 0.5;       % Current in Amperes for McEliece
current_aes = 0.1;      % Current in Amperes for AES-256
time_mc = 1.2e-3;       % Encryption time for McEliece in seconds
time_aes = 0.3e-3;      % Encryption time for AES-256 in seconds

% Energy calculations
energy_mc = voltage * current_mc * time_mc; % Energy for McEliece (Joules)
energy_aes = voltage * current_aes * time_aes; % Energy for AES-256 (Joules)

% Display results with better formatting
fprintf('Energy Consumption for McEliece: %.6f J\n', energy_mc);
fprintf('Energy Consumption for AES-256: %.6f J\n', energy_aes);
