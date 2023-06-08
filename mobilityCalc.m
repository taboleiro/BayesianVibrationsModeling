data = readtable("centerFreqResponse.csv");

Y_exp = abs(data.velocity + 1i*data.force);
freq = data.freq;

% Beam properties
beam = struct();
beam.length = 0.301;
beam.width = 0.026;
beam.thickness = 0.003;
beam.mass = 0.1877;

beam.massPerUnit = beam.mass / beam.length;
beam.volume = beam.length * beam.width * beam.thickness;
beam.I = beam.width * beam.thickness^3 / 12;

% Param initialization
E = 10e10;
rho = 8040;
eta = 0.007;
Y_calc = mobilityFuncModel(E, rho, eta, freq, beam);

% Results
figure(1)
plot(freq, 20*log10(Y_exp))
hold on
plot(freq, 20*log10(Y_calc),"Marker", ".")
xlabel("frequency / Hz")
ylabel("Mobility / dB")
legend("Experimental", "calculated")

function Y = mobilityFuncModel(E, rho, eta, freq, beam)
    % Calculates the mobility value based on the Young's modulus (E) and the frequency
    % Input: 
    %   E_dist   : Young's modulus distribution
    %   rho_dist : density distribution
    %   eta_dist : loss factor distribution
    %   freq     : frequency

    
    l = beam.length / 2;
    
    % Calculating the bending wave number
    w = 2 * pi * freq; % Angular frequency
    B = E * beam.I;
    complex_B = E * (1 + 1i * eta) * beam.I;
    massPerUnit = rho * beam.thickness * beam.width;
    cb = sqrt(w) * (B / massPerUnit)^(1/4); % Bending wave velocity
    
    kl = w ./ cb .* l; % Bending wave number
    complex_kl = kl .* (1 - 1i * eta / 4);
    N_l = cos(complex_kl) .* cosh(complex_kl) + 1;
    D_l = cos(complex_kl) .* sinh(complex_kl) + sin(complex_kl) .* cosh(complex_kl);
    
    Y = -(1i * l) ./ (2 * complex_kl .* sqrt(complex_B * massPerUnit)) .* N_l ./ D_l;
    Y = abs(Y);
end